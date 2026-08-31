# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Readers, filenames and thresholds every release check shares.

Extracted by #637 stage 3c so the checks below form a one-way graph: each
reads from here, and only ``publication`` reads from the checks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal, Mapping
import pandas as pd

#: ArviZ's default power-scaling flag threshold. A parameter is "sensitive" when
#: either its prior or its likelihood power-scaling statistic reaches this.
PSENSE_THRESHOLD = 0.05


#: Trace backing the natural-effect temporal-ordering sensitivity.  The fixed
#: basename is part of the release contract: a table is not independently
#: auditable when its posterior exists only in memory during the fit.
MEDIATION_T3_TRACE_FILENAME = "trace_mediation_t3_sensitivity.nc"


GROWTH_INFLUENCE_TRACE_FILENAME = "trace_growth_influence_sensitivity.nc"


#: Predeclared new-child predictive-adequacy floors for that design, pooled and per
#: outcome. The ordinary conditional check is saturated by construction and PSIS-LOO
#: is deliberately not computed, so the marginal check is the only one that can fail
#: — and "can fail" needs a stated threshold rather than a reader's judgement. Set
#: below nominal because the check is deliberately conservative (a redrawn residual
#: widens the interval), and breaching it *qualifies* a release rather than
#: withholding it: substantive misfit is information about the model, not evidence
#: that the sampler failed.
JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS: dict[int, float] = {50: 0.35, 90: 0.75}


TauSensitivityClass = Literal[
    "clear", "prior_data_conflict", "prior_dominant", "unavailable"
]


ReleaseStatus = Literal["release", "qualify", "withhold"]


#: Families the gate covers, each keyed to the term its causal headline rests on.
#: A family is here only if a randomised contrast identifies its headline — the
#: observational families report adjusted associations, which the reports already
#: label as such, and gating those on prior sensitivity would say nothing a reader
#: does not already know from the label.
GATED_KINDS = frozenset({"itt", "joint", "did", "gain_factors", "level_factors"})


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and abs(number) != float("inf") else None


def _read_csv(output_dir: str | Path, name: str, **kwargs: Any) -> pd.DataFrame | None:
    path = Path(output_dir) / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return None


def _config_name(output_dir: str | Path, model_id: str) -> str:
    """The ``-<config>`` suffix of a fit directory (``…-reporting`` -> ``reporting``)."""
    name = Path(output_dir).resolve().name
    prefix = f"{model_id}-"
    return name[len(prefix) :] if model_id and name.startswith(prefix) else ""


def _model_tier(config: Mapping[str, Any]) -> str:
    """Classify a fit for tiering purposes.

    ``adjusted_robustness`` is any ITT fit carrying a robustness adjustment set (the
    SES and general-ability comparators); ``off_grid`` is an outcome the standard
    44-cell sweep does not cover; everything else is ``primary``. Recorded on the
    decision even though the policy is currently uniform, so a graded policy needs no
    new plumbing and so the audit trail says which tier a fit was judged in.

    The ``adjusted_robustness`` test is deliberately ITT-only. Other families put
    confounders in ``adjust_for`` as part of their *primary* specification — the DAG
    adjustment set of a level- or gain-factor primary, not a robustness comparator —
    so keying the tier on the presence of an adjustment set alone labelled eight of
    the eleven level primaries as robustness comparators (#584 lower-severity 7).
    The withhold policy is uniform across tiers, so this corrects the audit label
    rather than any release decision.
    """
    from language_reading_predictors.statistical_models.sensitivity import (
        STANDARD_SENSITIVITY_OUTCOMES,
    )

    plan = config.get("resolved_run_plan") or {}
    if config.get("kind") == "itt" and (
        plan.get("adjust_for") or plan.get("adjustment")
    ):
        return "adjusted_robustness"
    if str(config.get("outcome_symbol") or "") not in STANDARD_SENSITIVITY_OUTCOMES:
        return "off_grid"
    return "primary"


def _load_config(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "config.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


RELEASE_DECISION_FILENAME = "release_decision.json"


PublicationStatus = Literal[
    "ok",
    "not_available",
    "inputs_unresolved",
    "gate_failed",
    "artifacts_incomplete",
    "robustness_unresolved",
]


ReleaseStage = Literal["inputs", "computation", "artifacts", "robustness"]


def _read_json(path: str | Path) -> tuple[Any, str | None]:
    """``(payload, error)`` — ``error`` names why the payload is unusable."""
    if not os.path.exists(path):
        return None, "missing"
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "unreadable"


def _stored_bool(value: Any) -> bool | None:
    """Parse a persisted Boolean without treating arbitrary truthy text as evidence."""

    normalised = str(value).strip().casefold()
    if normalised in {"true", "1", "yes"}:
        return True
    if normalised in {"false", "0", "no"}:
        return False
    return None


#: Binding fields a factorised joint contrast parent and its LKJ residual-correlation
#: companion must agree on before the pair counts as evidence about the *same*
#: estimand on the *same* rows (2026-08-23 joint audit, finding 2). Each entry is
#: ``(reader, human-readable description)``; the reader pulls the value out of a
#: stored ``config.json``. Comparing the resolved *plan* rather than the declared
#: settings is deliberate: the plan is what the fit actually ran.
_JOINT_PAIR_BINDING: tuple[tuple[str, Callable[[Mapping[str, Any]], Any]], ...] = (
    ("the ordered outcome list", lambda c: list(_plan(c).get("outcomes") or [])),
    (
        "the contrast direction",
        lambda c: [
            str((_plan(c).get("contrast") or {}).get("left") or ""),
            str((_plan(c).get("contrast") or {}).get("right") or ""),
        ],
    ),
    (
        "the fitted equation's precision terms",
        lambda c: [
            bool(_plan(c).get("use_cross_baselines")),
            bool(_plan(c).get("use_age_linear")),
            bool(_plan(c).get("use_age_gp")),
        ],
    ),
    ("the PSIS-LOO unit", lambda c: str(_plan(c).get("loo_unit") or "")),
    ("the input data checksum", lambda c: str(c.get("data_sha256") or "") or None),
    (
        "the fitted-row identity",
        lambda c: str((c.get("fitted_subject_identity") or {}).get("sha256") or "")
        or None,
    ),
    (
        "the fitted-data digest and observed denominators",
        lambda c: (
            [digest, (c.get("fitted_data_identity") or {}).get("observed")]
            if (digest := str((c.get("fitted_data_identity") or {}).get("digest") or ""))
            else None
        ),
    ),
    ("the sampling configuration", lambda c: c.get("sampling") or None),
    (
        "the source commit",
        lambda c: str(
            ((c.get("provenance") or {}).get("source") or {}).get("commit") or ""
        )
        or None,
    ),
)


def _plan(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The stored resolved run plan, or an empty mapping."""
    plan = config.get("resolved_run_plan")
    return plan if isinstance(plan, Mapping) else {}


#: Binding fields a within-child historical-joint fit and its wider-prior
#: sensitivity companion must agree on before the pair counts as evidence about the
#: *same* model under a *different* prior (#588 finding 5). Everything the fitted
#: equation and the analysis rows depend on, and every prior scale except the one
#: under test.
_HISTORICAL_JOINT_PRIOR_BINDING: tuple[
    tuple[str, Callable[[Mapping[str, Any]], Any]], ...
] = (
    ("the measure list", lambda c: list(_plan(c).get("measures") or [])),
    (
        "the analysis window",
        lambda c: [
            list(_plan(c).get("waves") or []),
            list(_plan(c).get("extension_waves") or []),
        ],
    ),
    ("the likelihood", lambda c: str(_plan(c).get("likelihood") or "")),
    (
        "the priors not under test",
        lambda c: [
            _plan(c).get("eta_prior_sigma"),
            _plan(c).get("sigma_subject_prior_sigma"),
            _plan(c).get("lkj_eta"),
            _plan(c).get("within_lkj_eta"),
        ],
    ),
    ("the input data checksum", lambda c: str(c.get("data_sha256") or "") or None),
    (
        "the fitted-row identity",
        lambda c: str((c.get("fitted_subject_identity") or {}).get("sha256") or "")
        or None,
    ),
)


#: How far a measure's power-scaling prior sensitivity may sit before the fit's own
#: diagnostics say the within-scale prior is doing the deciding. ArviZ's own flag
#: threshold; quoted rather than re-derived so the qualification and the psense
#: table cannot disagree.
_HISTORICAL_JOINT_PRIOR_SENSITIVE = PSENSE_THRESHOLD
