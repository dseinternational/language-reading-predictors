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
        lambda _ctx, *, draws, var_names=None: events.append(
            f"prior_predictive[{draws}]"
            if var_names is None
            else f"prior_predictive[{draws},{var_names}]"
        ),
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
        # ``fallback_var_names`` carries the curated list the ESS-evolution panel
        # falls back to when a family declares no causal term; recorded here so
        # the stage contract is pinned, not just tolerated.
        lambda _ctx, *, causal_term, include_loo_pit, fallback_var_names=None: (
            events.append(
                f"extended[{causal_term},loo_pit={include_loo_pit},"
                f"fallback={tuple(fallback_var_names or ())}]"
            )
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
    ctx = SimpleNamespace(lifecycle_stages=[])
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
    ctx = SimpleNamespace(lifecycle_stages=[])
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
    ctx = SimpleNamespace(lifecycle_stages=[])
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
    ctx = SimpleNamespace(lifecycle_stages=[])
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
    ctx = SimpleNamespace(lifecycle_stages=[])
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
        "extended[tau,loo_pit=True,fallback=('alpha', 'tau')]",
        "save_trace",
    ]


def test_run_primary_fit_honours_the_genuine_family_differences(monkeypatch):
    """No LOO (mediation), no extended term, distinct psense vars, no trace save —
    the plan expresses each difference without changing the shared order."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(lifecycle_stages=[])
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


def test_run_primary_fit_owns_the_late_families_post_trace_order(monkeypatch):
    """Overlay and forest after the trace, then power scaling — inside the runner.

    Six families used to declare ``psense_timing="family_tail"``, which did
    nothing, and call ``run_psense`` themselves afterwards. The runner could
    neither order the stage nor know it had happened, so the only check available
    was reading the six pipelines' source for the right call order (#637 stage 4).
    """
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(lifecycle_stages=[])
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=("alpha", "tau"),
            psense_timing="after_trace",
            psense_vars=("tau",),
            prepare_psense=lambda _ctx: events.append("prepare_psense"),
            after_trace_audit=lambda _ctx: events.append("overlay_and_forest"),
            extended_term="tau",
        ),
    )

    assert events[-5:] == [
        "extended[tau,loo_pit=True,fallback=('alpha', 'tau')]",
        "save_trace",
        "overlay_and_forest",
        "prepare_psense",
        "psense['tau']",
    ]
    assert ctx.lifecycle_stages[-3:] == [
        "save_trace",
        "after_trace_audit",
        "power_scaling",
    ]


def test_a_fit_with_nothing_to_power_scale_declares_it(monkeypatch):
    """``skip`` is a declaration, not an omission.

    A treated-only gain-factor variant has no focal term. Saying so means the
    absence of ``psense_summary.csv`` is a stated decision rather than a slot
    nobody happened to fill.
    """
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(lifecycle_stages=[])
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=("alpha",),
            psense_timing="skip",
            prepare_psense=lambda _ctx: events.append("must_not_prepare_psense"),
        ),
    )

    assert "must_not_prepare_psense" not in events
    assert not any(event.startswith("psense") for event in events)
    assert "power_scaling" not in ctx.lifecycle_stages


def test_power_scaling_runs_exactly_once_and_the_runner_knows_it(monkeypatch):
    """The property the escape hatch made uncheckable."""
    import pytest

    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(lifecycle_stages=[])
    _patch_primary_fit_diag(monkeypatch, events)

    for timing in ("before_ppc", "after_ppc", "before_trace", "after_trace"):
        runner.run_primary_fit(
            ctx,
            PrimaryFitPlan(diagnostic_vars=("alpha",), psense_timing=timing),
        )
        assert ctx.lifecycle_stages.count("power_scaling") == 1, timing
        assert [e for e in events if e.startswith("psense")] == ["psense['alpha']"]
        events.clear()

    # A family hook that reaches for power scaling itself is refused rather than
    # silently producing a second, differently-scoped sensitivity table.
    def _second_psense(_ctx):
        runner_psense = stages._diag.run_psense
        runner_psense(_ctx, var_names=["alpha"])

    with pytest.raises(RuntimeError, match="ran twice"):
        runner.run_primary_fit(
            ctx,
            PrimaryFitPlan(
                diagnostic_vars=("alpha",),
                psense_timing="after_trace",
                after_trace_audit=lambda c: c.lifecycle_stages.append("power_scaling"),
            ),
        )


def test_run_primary_fit_can_run_psense_after_ppc_and_return_the_gate(monkeypatch):
    """Adjusted RLI retains PPC-before-psense and consumes the one gate record."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(lifecycle_stages=[])
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
    ctx = SimpleNamespace(lifecycle_stages=[])
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=("alpha",),
            post_ppc_audit=lambda _ctx: events.append("post_ppc_audit"),
            # This test is about the audit's slot, not about sensitivity.
            psense_timing="skip",
        ),
    )

    assert events.index("save_ppc") < events.index("post_ppc_audit")
    assert events.index("post_ppc_audit") < events.index("Extended diagnostics")
    assert not any(event.startswith("psense") for event in events)


def test_run_primary_fit_orders_exceptional_phase_hooks(monkeypatch):
    """Custom prior, stitched LOO, PPC and diagnostics stay in named phase slots."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(lifecycle_stages=[])
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=("alpha",),
            summary_header="Anchor summary",
            extended_header="Anchor diagnostics",
            prior_predictive_var_names=("z_a", "z_b"),
            plot_prior_predictive=lambda _ctx: events.append("plot_prior"),
            post_sampling_audit=lambda _ctx: events.append("stitched_loo"),
            custom_posterior_predictive=lambda _ctx: events.append("custom_ppc"),
            psense_timing="before_trace",
            post_gate_audit=lambda _ctx, _gate: events.append("post_gate_audit"),
            post_extended_audit=lambda _ctx: events.append("custom_diagnostics"),
            compute_loo=False,
            include_loo_pit=False,
        ),
    )

    assert events == [
        "Prior predictive",
        "prior_predictive[1000,['z_a', 'z_b']]",
        "plot_prior",
        "Sampling posterior (nutpie)",
        "sample",
        "stitched_loo",
        "Anchor summary",
        "summary['alpha']",
        "Posterior predictive",
        "custom_ppc",
        "Anchor diagnostics",
        "gate['alpha']",
        "post_gate_audit",
        "extended[None,loo_pit=False,fallback=('alpha',)]",
        "custom_diagnostics",
        "psense['alpha']",
        "save_trace",
    ]


def test_run_primary_fit_supports_termless_extended_diagnostics(monkeypatch):
    """Associational families run the extended block without a focal term."""
    events = []
    runner = _stage_runner(events)
    ctx = SimpleNamespace(lifecycle_stages=[])
    _patch_primary_fit_diag(monkeypatch, events)

    runner.run_primary_fit(
        ctx,
        PrimaryFitPlan(diagnostic_vars=("alpha",), extended_term=None),
    )

    assert "extended[None,loo_pit=True,fallback=('alpha',)]" in events


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
