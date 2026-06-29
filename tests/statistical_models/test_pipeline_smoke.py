# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke tests for the shared statistical-pipeline phases (#82).

The ``fit_*`` pipelines in ``statistical_models.pipeline`` were previously
uncovered — the gap behind the #78 drift, where one pipeline diverged from the
others because the scaffold was copy-pasted. #82 centralised the byte-identical
phases into four helpers; these tests lock that down:

* the shared phase helpers exist;
* every ``fit_*`` routes its build-attach / sampling+LOO / posterior-predictive
  / report phases **through** those helpers rather than re-inlining them;
* each shared phase body lives in exactly **one** place (its helper); and
* a real minimal end-to-end ITT fit still runs and emits its artefacts.

The structural checks are fast and need no sampling; the final test runs one
genuine fit at a tiny sampling configuration as a runtime tripwire.
"""

import inspect
import os

import pytest

from language_reading_predictors.statistical_models import pipeline as P

# Every public pipeline entry point and the kind it serves.
FIT_FUNCTIONS = [
    "fit_itt",
    "fit_joint",
    "fit_did",
    "fit_mechanism",
    "fit_dose_response",
    "fit_mediation",
    "fit_gain_factors",
    "fit_level_factors",
    "fit_aligned",
    "fit_mediation_multi",
    "fit_adjusted",
    "fit_lcsm",
]

# The shared phase helpers introduced in #82.
SHARED_HELPERS = [
    "_attach_built",
    "_run_sampling_and_loo",
    "_run_ppc",
    "_finalize_report",
]


def test_shared_phase_helpers_exist():
    for name in SHARED_HELPERS:
        assert callable(getattr(P, name, None)), f"missing shared helper {name}"


@pytest.mark.parametrize("name", FIT_FUNCTIONS)
def test_pipeline_kind_has_fit_function(name):
    assert callable(getattr(P, name, None)), f"missing pipeline entry point {name}"


@pytest.mark.parametrize("name", FIT_FUNCTIONS)
def test_fit_routes_through_shared_phases(name):
    """Each pipeline must use the shared helpers, not a re-copied scaffold (#82).

    Guards against the #78 failure mode: a pipeline (existing or newly added)
    drifting because it inlines the build-attach / sampling / PPC / report
    blocks instead of calling the centralised helpers.
    """
    src = inspect.getsource(getattr(P, name))
    assert "_attach_built(ctx, built)" in src, f"{name}: not using _attach_built"
    assert "_run_sampling_and_loo(ctx" in src, f"{name}: not using _run_sampling_and_loo"
    assert "_run_ppc(ctx" in src, f"{name}: not using _run_ppc"
    assert "return _finalize_report(ctx)" in src, f"{name}: not using _finalize_report"


def test_shared_phase_bodies_are_centralised():
    """Each shared phase body must appear exactly once — inside its helper.

    A count > 1 means a pipeline has re-inlined a phase the helpers own, which
    is exactly the duplication #82 removed.
    """
    src = inspect.getsource(P)
    once = {
        "ctx.model = built.model": "build-attach (_attach_built)",
        "_diag.sample_posterior(ctx)": "posterior sampling (_run_sampling_and_loo)",
        "_diag.compute_log_likelihood_and_loo(ctx)": "LOO-PSIS (_run_sampling_and_loo)",
        "_diag.sample_posterior_predictive(ctx, var_names=": "PPC draw (_run_ppc)",
        "_copy_report_template(ctx)": "report template copy (_finalize_report)",
        "_print_footer(ctx)": "footer (_finalize_report)",
    }
    for block, where in once.items():
        assert src.count(block) == 1, (
            f"{where!r} appears {src.count(block)}x; it must live only in its helper"
        )


def test_itt_pipeline_end_to_end_smoke(tmp_path, monkeypatch):
    """One real minimal ITT fit through the refactored pipeline.

    Exercises ``_attach_built`` -> ``_run_sampling_and_loo`` (with LOO) ->
    ``_run_ppc`` -> ``_finalize_report`` end to end at a tiny sampling config,
    writing to a temporary output root. The runtime tripwire for #82.
    """
    from dse_research_utils.statistics.models.sampling import SamplingConfiguration

    from language_reading_predictors.statistical_models import context as ctxmod
    from language_reading_predictors.statistical_models import lrpitt07

    tiny = SamplingConfiguration(
        draws=100, tune=100, chains=2, cores=1, target_accept=0.9, random_seed=47
    )
    monkeypatch.setattr(
        ctxmod._sampling, "get_sampling_configuration", lambda *a, **k: tiny
    )
    monkeypatch.setattr(ctxmod._env, "STAT_OUTPUT_DIR", str(tmp_path))

    ctx = lrpitt07.fit("dev")

    assert isinstance(ctx, ctxmod.StatisticalFitContext)
    assert ctx.trace is not None and "tau" in ctx.trace.posterior  # estimand sampled
    assert ctx.loo is not None  # ITT runs the LOO branch
    assert "tau_summary" in ctx.tables
    assert os.path.isfile(os.path.join(ctx.output_dir, "tau_summary.csv"))
