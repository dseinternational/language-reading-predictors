# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The automatic sampling-quality gate: what it checks, and how it reads.

R-hat <= 1.01, bulk and tail ESS >= 400, BFMI >= 0.3 and zero divergences. The
verdict outranks every other publication consideration, so a great many modules
read it: the release evaluator, the key-findings box, the report badge, the
blending-pair and influence checks, and several family pipelines.

It lives here rather than in ``key_findings`` or ``release`` because those two
imported each other to reach it (#637 stage 3): ``release`` needed the gate reader
from ``reporting`` while ``key_findings`` needed the release decision, and both
edges were function-local imports written to hide the cycle.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


# Human-readable names for the convergence-gate checks (the gate-failed banner).
_KF_CHECK_LABELS = {
    "rhat": "R-hat",
    "ess": "effective sample size",
    "divergences": "divergent transitions",
    "bfmi": "sampling energy (BFMI)",
}


_KF_REQUIRED_CHECKS = tuple(_KF_CHECK_LABELS)


_KF_RHAT_MAX = 1.01


_KF_ESS_MIN = 400.0


_KF_BFMI_MIN = 0.3


def _raw_convergence_gate_checks(diag_summary: Mapping) -> dict[str, bool] | None:
    """Recompute the fixed house gate from unrounded stored measurements.

    The stored ``passed`` and ``checks`` fields are useful audit records, but they
    are mutable summaries. Scientific-result consumers therefore require the raw
    measurements too and independently apply the house thresholds. Missing or
    non-numeric fields return ``None`` so the caller fails closed as incomplete.
    """

    required = ("divergences", "max_rhat", "min_ess", "bfmi_per_chain")
    if any(name not in diag_summary for name in required):
        return None
    try:
        if isinstance(diag_summary["divergences"], bool):
            return None
        divergences = float(diag_summary["divergences"])
        max_rhat = float(diag_summary["max_rhat"])
        min_ess = float(diag_summary["min_ess"])
    except (TypeError, ValueError):
        return None

    bfmi_raw = diag_summary["bfmi_per_chain"]
    if isinstance(bfmi_raw, (str, bytes, Mapping)):
        return None
    try:
        bfmi = np.asarray(list(bfmi_raw), dtype=float)
    except (TypeError, ValueError):
        return None
    if bfmi.size == 0:
        return None

    return {
        "rhat": bool(np.isfinite(max_rhat) and max_rhat <= _KF_RHAT_MAX),
        "ess": bool(np.isfinite(min_ess) and min_ess >= _KF_ESS_MIN),
        "divergences": bool(np.isfinite(divergences) and divergences == 0),
        "bfmi": bool(np.isfinite(bfmi).all() and np.all(bfmi >= _KF_BFMI_MIN)),
    }


def convergence_gate_failures(diag_summary: Mapping | None) -> list[str]:
    """Return readable failing checks from a sampling-quality gate payload.

    ``passed`` is the measured verdict written by ``dse_research_utils``. The
    required per-check values must agree with it; an absent or internally
    inconsistent payload fails closed.

    The automatic gate deliberately has no model-spec waiver. A future qualified
    divergence-only result must follow the trace-bound review policy in ``METHODS.md``;
    until a reviewed qualification artefact and verifier exist, every failed
    ``diagnostics_summary.json`` fails closed here.
    """
    if not isinstance(diag_summary, Mapping):
        return ["convergence summary incomplete"]

    checks = diag_summary.get("checks")
    if not isinstance(checks, Mapping) or any(
        name not in checks for name in _KF_REQUIRED_CHECKS
    ):
        return ["convergence summary incomplete"]

    raw_checks = _raw_convergence_gate_checks(diag_summary)
    if raw_checks is None:
        return ["convergence summary incomplete"]

    failing_names = [
        name
        for name in _KF_REQUIRED_CHECKS
        if checks.get(name) is not True or raw_checks[name] is not True
    ]
    failing_names.extend(
        str(name)
        for name, ok in checks.items()
        if name not in _KF_REQUIRED_CHECKS and ok is not True
    )
    if failing_names:
        return [_KF_CHECK_LABELS.get(n, n) for n in failing_names]
    if diag_summary.get("passed") is not True:
        return ["convergence summary incomplete"]
    return []


def convergence_gate_clean_passed(diag_summary: Mapping | None) -> bool:
    """Return whether the complete automatic sampling-quality gate passed cleanly."""
    return not convergence_gate_failures(diag_summary)


def convergence_gate_badge_markdown(
    diag_summary: Mapping | None,
    _legacy_gate_exception: Mapping | None = None,
) -> str:
    """Render the compact gate badge shown before report findings (#321).

    The automatic renderer has two states: a clean pass (green ``tip``) or a failure /
    unavailable gate (red ``important``, findings withheld). The optional second
    argument keeps older copied report partials callable but is deliberately ignored:
    a model-spec exception can no longer alter the verdict. A divergence-qualified
    result is not an automatic pass; the policy requires a trace-bound reviewed
    artefact and a separate verifier before a third, amber state may be implemented.
    The full numerical banner remains under Technical checks.
    """
    failing = convergence_gate_failures(diag_summary)
    if failing:
        failed_text = ", ".join(failing)
        return (
            '::: {.callout-important title="Sampling-quality gate: failed"}\n\n'
            f"**FAIL** — Sampling-quality checks failed: {failed_text}. Findings are "
            "withheld; review Technical checks before interpreting any estimate.\n\n:::"
        )
    return (
        '::: {.callout-tip title="Sampling-quality gate: passed"}\n\n'
        "**PASS** — All sampling-quality checks passed; details are under "
        "Technical checks.\n\n:::"
    )
