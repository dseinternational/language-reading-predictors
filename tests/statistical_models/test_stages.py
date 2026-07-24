# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the shared statistical-model execution stages."""

from __future__ import annotations

from types import SimpleNamespace

from language_reading_predictors.statistical_models import stages
from language_reading_predictors.statistical_models.stages import (
    SharedFitStages,
    StageHooks,
)


def _stage_runner(events):
    def hook(name):
        return lambda *_args, **_kwargs: events.append(name)

    return SharedFitStages(
        StageHooks(
            emit_priors=hook("priors"),
            save_ppc=hook("save_ppc"),
            write_loo_influence=hook("influence"),
            print_loo_row=hook("loo_row"),
            copy_report_template=hook("copy_report"),
            publish_output=hook("publish"),
            print_footer=hook("footer"),
        )
    )


def test_attach_stage_sets_the_built_contract_before_priors():
    events = []
    ctx = SimpleNamespace(model=None, model_vars=None, prepared=None)
    built = SimpleNamespace(model="model", variables={"tau": 1}, prepared="data")

    _stage_runner(events).attach_built(ctx, built)

    assert (ctx.model, ctx.model_vars, ctx.prepared) == (
        "model",
        {"tau": 1},
        "data",
    )
    assert events == ["priors"]


def test_sampling_stage_keeps_sampling_loo_reporting_order(monkeypatch):
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    monkeypatch.setattr(stages, "section_header", lambda title: events.append(title))
    monkeypatch.setattr(
        stages._diag, "sample_posterior", lambda _ctx: events.append("sample")
    )
    monkeypatch.setattr(
        stages._diag,
        "compute_log_likelihood_and_loo",
        lambda _ctx: events.append("loo"),
    )
    monkeypatch.setattr(
        stages._report,
        "write_loo_summary",
        lambda _ctx: events.append("loo_summary"),
    )

    runner.sample_and_loo(ctx)

    assert events == [
        "Sampling posterior (nutpie)",
        "sample",
        "LOO-PSIS",
        "loo",
        "loo_summary",
        "influence",
        "loo_row",
    ]


def test_sample_and_loo_skips_the_loo_block_when_disabled(monkeypatch):
    """compute_loo=False is the mediation-family path (no ordinary PSIS-LOO); the
    shared lifecycle must sample but skip the whole LOO/report/influence block. A
    refactor that centralises the lifecycle must preserve this genuine difference."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    monkeypatch.setattr(stages, "section_header", lambda title: events.append(title))
    monkeypatch.setattr(
        stages._diag, "sample_posterior", lambda _ctx: events.append("sample")
    )

    def _fail(_ctx):
        raise AssertionError("LOO must not run when compute_loo=False")

    monkeypatch.setattr(stages._diag, "compute_log_likelihood_and_loo", _fail)
    monkeypatch.setattr(stages._report, "write_loo_summary", _fail)

    runner.sample_and_loo(ctx, compute_loo=False)

    assert events == ["Sampling posterior (nutpie)", "sample"]


def test_posterior_predictive_defaults_to_the_y_post_node(monkeypatch):
    """With no explicit var_names the shared PPC stage draws (and saves the primary
    node as) ``y_post`` — the default observation node for the count families."""
    sampled = []
    saved = []
    ctx = SimpleNamespace()
    monkeypatch.setattr(stages, "section_header", lambda _title: None)
    monkeypatch.setattr(
        stages._diag,
        "sample_posterior_predictive",
        lambda _ctx, *, var_names: sampled.append(var_names),
    )
    runner = SharedFitStages(
        StageHooks(
            emit_priors=lambda _ctx: None,
            save_ppc=lambda _ctx, *, primary_node: saved.append(primary_node),
            write_loo_influence=lambda _ctx: None,
            print_loo_row=lambda _ctx: None,
            copy_report_template=lambda _ctx: None,
            publish_output=lambda _ctx: None,
            print_footer=lambda _ctx: None,
        )
    )

    runner.posterior_predictive(ctx)

    assert sampled == [["y_post"]]
    assert saved == ["y_post"]


def test_posterior_predictive_uses_the_last_requested_node(monkeypatch):
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    sampled = []
    saved = []
    monkeypatch.setattr(stages, "section_header", lambda title: events.append(title))
    monkeypatch.setattr(
        stages._diag,
        "sample_posterior_predictive",
        lambda _ctx, *, var_names: sampled.append(var_names),
    )
    runner = SharedFitStages(
        StageHooks(
            emit_priors=lambda _ctx: None,
            save_ppc=lambda _ctx, *, primary_node: saved.append(primary_node),
            write_loo_influence=lambda _ctx: None,
            print_loo_row=lambda _ctx: None,
            copy_report_template=lambda _ctx: None,
            publish_output=lambda _ctx: None,
            print_footer=lambda _ctx: None,
        )
    )

    runner.posterior_predictive(ctx, var_names=["mediator_post", "y_post"])

    assert sampled == [["mediator_post", "y_post"]]
    assert saved == ["y_post"]


def test_metadata_and_report_finalization_are_shared(monkeypatch):
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(output_dir="output")
    metadata = []
    monkeypatch.setattr(
        stages._report,
        "write_run_metadata",
        lambda context, *, extra: metadata.append((context, extra)),
    )
    monkeypatch.setattr(
        stages._report,
        "generate_key_findings",
        lambda _output: {"status": "ok", "sentences": ["one"]},
    )
    monkeypatch.setattr(stages, "section_header", lambda title: events.append(title))

    runner.write_metadata(ctx, extra={"family": "example"})
    returned = runner.finalize_report(ctx)

    assert metadata == [(ctx, {"family": "example"})]
    assert returned is ctx
    assert events == ["Report", "copy_report", "publish", "footer"]
