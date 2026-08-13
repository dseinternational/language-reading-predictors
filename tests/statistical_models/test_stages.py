# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the shared statistical-model execution stages."""

from __future__ import annotations

import json
from types import SimpleNamespace

from language_reading_predictors.statistical_models import stages
from language_reading_predictors.statistical_models.stages import (
    PrimaryFitPlan,
    SharedFitStages,
    StageHooks,
)


def _patch_primary_fit_diag(monkeypatch, events):
    """Route every diagnostics call of the primary-fit lifecycle into ``events``."""
    monkeypatch.setattr(stages, "section_header", lambda title: events.append(title))
    monkeypatch.setattr(
        stages._diag,
        "run_prior_predictive",
        lambda _ctx, *, draws: events.append(f"prior_predictive[{draws}]"),
    )
    monkeypatch.setattr(
        stages._diag, "sample_posterior", lambda _ctx: events.append("sample")
    )
    monkeypatch.setattr(
        stages._diag,
        "compute_log_likelihood_and_loo",
        lambda _ctx: events.append("loo"),
    )
    monkeypatch.setattr(
        stages._report, "write_loo_summary", lambda _ctx: events.append("loo_summary")
    )
    monkeypatch.setattr(
        stages._diag,
        "summary_diagnostics",
        lambda _ctx, *, var_names: events.append(f"summary{var_names}"),
    )
    monkeypatch.setattr(
        stages._diag,
        "run_psense",
        lambda _ctx, *, var_names: events.append(f"psense{var_names}"),
    )
    monkeypatch.setattr(
        stages._diag,
        "sample_posterior_predictive",
        lambda _ctx, *, var_names: events.append(f"ppc_sample{var_names}"),
    )
    monkeypatch.setattr(
        stages._diag,
        "write_diagnostics_summary",
        lambda _ctx, *, var_names: events.append(f"gate{var_names}"),
    )
    monkeypatch.setattr(
        stages._diag,
        "run_extended_diagnostics",
        lambda _ctx, *, causal_term, include_loo_pit: events.append(
            f"extended[{causal_term},loo_pit={include_loo_pit}]"
        ),
    )
    monkeypatch.setattr(
        stages._diag, "save_trace", lambda _ctx: events.append("save_trace")
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
    ctx = SimpleNamespace(model=None, prepared=None)
    built = SimpleNamespace(model="model", prepared="data")

    _stage_runner(events).attach_built(ctx, built)

    assert (ctx.model, ctx.prepared) == ("model", "data")
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


def test_run_primary_fit_owns_the_invariant_sequence(monkeypatch):
    """The primary-fit lifecycle order, expressed once (#394 acceptance criterion).

    Prior predictive (with the family's figure) -> sampling -> LOO -> summary
    diagnostics -> power-scaling sensitivity -> posterior predictive -> the
    all-free-variable convergence gate -> extended diagnostics -> trace
    persistence.
    """
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    _patch_primary_fit_diag(monkeypatch, events)

    plan = PrimaryFitPlan(
        diagnostic_vars=("alpha", "tau"),
        ppc_var_names=("y_event",),
        plot_prior_predictive=lambda _ctx: events.append("plot_prior_predictive"),
        extended_term="tau",
    )
    runner.run_primary_fit(ctx, plan)

    assert events == [
        "Prior predictive",
        "prior_predictive[1000]",
        "plot_prior_predictive",
        "Sampling posterior (nutpie)",
        "sample",
        "LOO-PSIS",
        "loo",
        "loo_summary",
        "influence",
        "loo_row",
        "Summary diagnostics",
        "summary['alpha', 'tau']",
        "psense['alpha', 'tau']",
        "Posterior predictive",
        "ppc_sample['y_event']",
        "save_ppc",
        "Extended diagnostics",
        "gate['alpha', 'tau']",
        "extended[tau,loo_pit=True]",
        "save_trace",
    ]


def test_run_primary_fit_honours_the_genuine_family_differences(monkeypatch):
    """No LOO (mediation), no extended term, distinct psense vars, no trace save —
    the plan expresses each difference without changing the shared order."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    _patch_primary_fit_diag(monkeypatch, events)

    plan = PrimaryFitPlan(
        diagnostic_vars=("alpha",),
        psense_vars=("tau",),
        prepare_psense=lambda _ctx: events.append("prepare_psense"),
        compute_loo=False,
        extended_term=None,
        run_extended=False,
        save_trace=False,
    )
    runner.run_primary_fit(ctx, plan)

    assert events == [
        "Prior predictive",
        "prior_predictive[1000]",
        "Sampling posterior (nutpie)",
        "sample",
        "Summary diagnostics",
        "summary['alpha']",
        "prepare_psense",
        "psense['tau']",
        "Posterior predictive",
        "ppc_sample['y_post']",
        "save_ppc",
        "Extended diagnostics",
        "gate['alpha']",
    ]


def test_run_primary_fit_can_leave_late_psense_to_the_family(monkeypatch):
    """Late-sensitivity families must reach trace persistence without power scaling.

    Their family pipeline then writes its established post-trace overlay and
    forest before calling ``run_psense``. This opt-out preserves that artefact
    order while the invariant lifecycle moves to the shared runner.
    """
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=("alpha", "tau"),
            psense_timing="family_tail",
            prepare_psense=lambda _ctx: events.append("must_not_prepare_psense"),
            extended_term="tau",
        ),
    )

    assert "must_not_prepare_psense" not in events
    assert not any(event.startswith("psense") for event in events)
    assert events[-3:] == [
        "gate['alpha', 'tau']",
        "extended[tau,loo_pit=True]",
        "save_trace",
    ]


def test_run_primary_fit_can_run_psense_after_ppc_and_return_the_gate(monkeypatch):
    """Adjusted RLI retains PPC-before-psense and consumes the one gate record."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    _patch_primary_fit_diag(monkeypatch, events)
    gate = {"passed": True, "checks": {"rhat": True}}
    monkeypatch.setattr(
        stages._diag,
        "write_diagnostics_summary",
        lambda _ctx, *, var_names: (events.append(f"gate{var_names}"), gate)[1],
    )

    returned = runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=("alpha", "beta"),
            psense_timing="after_ppc",
        ),
    )

    assert returned is gate
    assert events.index("save_ppc") < events.index("psense['alpha', 'beta']")
    assert events.index("psense['alpha', 'beta']") < events.index(
        "Extended diagnostics"
    )


def test_run_primary_fit_runs_post_ppc_audit_before_the_gate(monkeypatch):
    """DiD cell calibration is part of PPC and must precede the convergence gate."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=("alpha",),
            post_ppc_audit=lambda _ctx: events.append("post_ppc_audit"),
            psense_timing="family_tail",
        ),
    )

    assert events.index("save_ppc") < events.index("post_ppc_audit")
    assert events.index("post_ppc_audit") < events.index("Extended diagnostics")
    assert not any(event.startswith("psense") for event in events)


def test_run_primary_fit_supports_termless_extended_diagnostics(monkeypatch):
    """Associational families run the extended block without a focal term."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace()
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(diagnostic_vars=("alpha",), extended_term=None),
    )

    assert "extended[None,loo_pit=True]" in events


def test_metadata_and_report_finalization_are_shared(monkeypatch, tmp_path):
    events = []
    runner = _stage_runner(events)
    # A real (temporary) directory: finalisation now scans the output directory
    # to write the artefact manifest, so the context must not point at the
    # repository's real output root.
    ctx = SimpleNamespace(output_dir=str(tmp_path))
    metadata = []
    monkeypatch.setattr(
        stages._report,
        "write_run_metadata",
        lambda context, *, extra: metadata.append((context, extra)),
    )
    # The findings generator now *receives* the release decision rather than
    # making one (#394 design point 3), so the stub records what it was handed.
    findings_calls = []

    def _fake_findings(output, *, decision=None):
        findings_calls.append((output, decision))
        events.append("key_findings")
        return {"status": "ok", "sentences": ["one"]}

    monkeypatch.setattr(stages._report, "generate_key_findings", _fake_findings)
    monkeypatch.setattr(stages, "section_header", lambda title: events.append(title))

    runner.write_metadata(ctx, extra={"family": "example"})
    returned = runner.finalize_report(ctx)

    assert metadata == [(ctx, {"family": "example"})]
    assert returned is ctx
    assert events == ["Report", "key_findings", "copy_report", "publish", "footer"]
    # The manifest is written between the report copy and publication (#394).
    assert (tmp_path / "artifact_manifest.json").exists()
    # The release decision is settled and on disk *before* the findings that
    # follow from it — the acceptance criterion, asserted as an ordering.
    assert (tmp_path / "release_decision.json").exists()
    (_output, decision), = findings_calls
    assert decision is not None and decision.status == "not_available"
    written = json.loads((tmp_path / "release_decision.json").read_text())
    assert written["status"] == decision.status
    assert written["publishable"] is False
