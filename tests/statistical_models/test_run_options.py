# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for per-invocation statistical-model run options."""

from __future__ import annotations

import pytest

from language_reading_predictors.statistical_models import context
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.run_options import (
    StatisticalRunOptions,
    current_run_options,
    use_run_options,
)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_target_accept_override_must_be_a_probability(value):
    with pytest.raises(ValueError, match="open interval"):
        StatisticalRunOptions(target_accept=value)


def test_randomised_archive_option_must_name_a_path():
    with pytest.raises(ValueError, match="non-empty path"):
        StatisticalRunOptions(rli_randomised_archive="   ")

    options = StatisticalRunOptions(rli_randomised_archive="/tmp/rli.csv")
    assert options.rli_randomised_archive == "/tmp/rli.csv"


def test_run_options_are_nested_and_restored():
    outer = StatisticalRunOptions(target_accept=0.91)
    inner = StatisticalRunOptions(target_accept=0.97)

    assert current_run_options() == StatisticalRunOptions()
    with use_run_options(outer):
        assert current_run_options() is outer
        with use_run_options(inner):
            assert current_run_options() is inner
        assert current_run_options() is outer
    assert current_run_options() == StatisticalRunOptions()


def test_cli_override_is_scoped_and_outranks_model_default(tmp_path, monkeypatch):
    monkeypatch.setattr(context._env, "init_plotting", lambda: None)
    monkeypatch.setattr(context._paths, "stat_dir", lambda: tmp_path)
    original_getter = context._sampling.get_sampling_configuration
    spec = ModelSpec(
        model_id="lrp-rli-example-001",
        kind="example",
        title="Run-options test",
        extra={"target_accept": 0.999},
    )

    with use_run_options(StatisticalRunOptions(target_accept=0.93)):
        fit_context = context.make_context(spec)

    assert context._sampling.get_sampling_configuration is original_getter
    assert fit_context.sampling.target_accept == 0.93
    assert fit_context.run_options.target_accept == 0.93
    assert current_run_options() == StatisticalRunOptions()


def test_model_default_outranks_sampling_preset_without_cli_override(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(context._env, "init_plotting", lambda: None)
    monkeypatch.setattr(context._paths, "stat_dir", lambda: tmp_path)
    spec = ModelSpec(
        model_id="lrp-rli-example-002",
        kind="example",
        title="Model-default test",
        extra={"target_accept": 0.97},
    )

    fit_context = context.make_context(spec)

    assert fit_context.sampling.target_accept == 0.97


def test_spec_target_accept_is_honoured_for_every_family(tmp_path, monkeypatch):
    """A spec declaration must bind regardless of which fit function runs.

    ``spec.extra['target_accept']`` used to be applied by a pipeline helper that only
    six family entry points called, so the same declaration on a dose-response, DiD or
    ITT spec was accepted and then silently ignored — the fit sampled at the preset
    while the module claimed otherwise. Resolution now lives in ``make_context``, which
    every family goes through.
    """
    monkeypatch.setattr(context._env, "init_plotting", lambda: None)
    monkeypatch.setattr(context._paths, "stat_dir", lambda: tmp_path)

    for kind in ("dose_response", "did", "itt", "gain_factors", "mechanism"):
        spec = ModelSpec(
            model_id=f"lrp-rli-example-{kind}",
            kind=kind,
            title=f"{kind} spec-default test",
            extra={"target_accept": 0.985},
        )
        assert context.make_context(spec).sampling.target_accept == 0.985, kind


def test_spec_target_accept_must_be_a_probability(tmp_path, monkeypatch):
    monkeypatch.setattr(context._env, "init_plotting", lambda: None)
    monkeypatch.setattr(context._paths, "stat_dir", lambda: tmp_path)
    spec = ModelSpec(
        model_id="lrp-rli-example-003",
        kind="example",
        title="Invalid spec default",
        extra={"target_accept": 1.0},
    )

    with pytest.raises(ValueError, match="open interval"):
        context.make_context(spec)
