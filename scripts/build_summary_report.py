# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the plain-language summary report's results file and figures.

The report at ``docs/summary-report/`` is written for teachers and researchers
rather than for readers of the per-model pages, but it must obey the same rule
as the integrated report: **no number is hand-entered**. This script reads the
fitted artefacts under ``output/statistical_models/models/<id>-<config>/`` and
``output/models/<id>/`` and writes, to ``output/summary_report/``:

* ``results.json`` — every quantity the report quotes, with the model id it came
  from, so any sentence can be traced back to one fit;
* one figure per file (PNG + SVG + the CSV behind it), following the project's
  individual-figure policy.

Usage::

    python scripts/build_summary_report.py                 # reporting config
    python scripts/build_summary_report.py --config rep-lite

Quantities are taken from each family's own published summary CSV rather than
recomputed from traces, so the report cannot drift from the model pages.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import dse_research_utils.environment.setup as setup  # noqa: E402

from language_reading_predictors import paths  # noqa: E402
from language_reading_predictors.figure_io import save_styled_figure  # noqa: E402
from language_reading_predictors.statistical_models.measures import MEASURES  # noqa: E402


# --- Which fit answers which question -------------------------------------
# Curated rather than inferred: the report quotes one primary fit per outcome
# per family, and an automatic rule would silently pick up a companion or a
# moderation variant when the registry grows.

ITT_PRIMARY = {
    "L": "lrp-rli-itt-007",
    "W": "lrp-rli-itt-010",
    "TE": "lrp-rli-itt-002",
    "TR": "lrp-rli-itt-001",
    "B": "lrp-rli-itt-008",
    "UR": "lrp-rli-itt-003",
    "UE": "lrp-rli-itt-004",
    "F": "lrp-rli-itt-025",
    "EG": "lrp-rli-itt-030",
    "EI": "lrp-rli-itt-029",
    "T": "lrp-rli-itt-026",
    "R": "lrp-rli-itt-005",
    "E": "lrp-rli-itt-006",
    "P": "lrp-rli-itt-009",
    "N": "lrp-rli-itt-011",
}

DID_PRIMARY = {
    "W": "lrp-rli-did-001",
    "L": "lrp-rli-did-002",
    "B": "lrp-rli-did-003",
    "TE": "lrp-rli-did-004",
    "R": "lrp-rli-did-005",
    "TR": "lrp-rli-did-008",
    "E": "lrp-rli-did-009",
    "F": "lrp-rli-did-010",
    "P": "lrp-rli-did-011",
    "N": "lrp-rli-did-012",
    "EI": "lrp-rli-did-014",
    "EG": "lrp-rli-did-015",
}

GF_PRIMARY = {
    "W": "lrp-rli-gf-001",
    "R": "lrp-rli-gf-002",
    "E": "lrp-rli-gf-003",
    "L": "lrp-rli-gf-004",
    "P": "lrp-rli-gf-005",
    "B": "lrp-rli-gf-006",
    "F": "lrp-rli-gf-007",
    "T": "lrp-rli-gf-008",
    "TR": "lrp-rli-gf-009",
    "TE": "lrp-rli-gf-010",
    "N": "lrp-rli-gf-011",
}

LF_PRIMARY = {
    "W": "lrp-rli-lf-001",
    "R": "lrp-rli-lf-002",
    "E": "lrp-rli-lf-003",
    "L": "lrp-rli-lf-004",
    "P": "lrp-rli-lf-005",
    "B": "lrp-rli-lf-006",
    "F": "lrp-rli-lf-007",
    "T": "lrp-rli-lf-008",
    "TR": "lrp-rli-lf-009",
    "TE": "lrp-rli-lf-010",
    "N": "lrp-rli-lf-011",
}

# Order the effects figure and table by what the intervention taught directly,
# then what it might transfer to, then the two floored measures.
OUTCOME_ORDER = ["L", "W", "TE", "TR", "B", "UR", "UE", "F", "EG", "T", "EI", "R", "E", "P", "N"]

WORD_READING_ROBUSTNESS = [
    ("lrp-rli-itt-024", "Adjusted for non-verbal ability"),
    ("lrp-rli-itt-013", "Adjusted for family background (33 children)"),
    ("lrp-rli-itt-014", "Same 33 children, unadjusted"),
    ("lrp-rli-itt-027", "Adjusted for site"),
]

MEDIATION_ROUTES = [
    ("lrp-rli-med-059", "Letter-sound knowledge"),
    ("lrp-rli-med-074", "Nonword decoding"),
    ("lrp-rli-med-068", "Taught expressive vocabulary"),
    ("lrp-rli-med-080", "Taught receptive vocabulary"),
    ("lrp-rli-med-079", "Expressive grammar (negative control)"),
]

# Skill-to-skill models whose outcome is word reading. The first three are the
# comparable trio the cross-model comparison puts on one per-SD scale; the rest
# are quoted individually on their own item scale.
MECHANISM_SLOPES = [
    ("lrp-rli-mech-058", "Letter-sound knowledge"),
    ("lrp-rli-mech-057", "Expressive vocabulary"),
    ("lrp-rli-mech-056", "Receptive vocabulary"),
    ("lrp-rli-mech-089", "Taught expressive vocabulary"),
    ("lrp-rli-mech-088", "Taught receptive vocabulary"),
    ("lrp-rli-mech-090", "Phonological memory (word/nonword repetition)"),
]

# Outcomes whose modelled count is not the scale a reader knows the test by.
# APT information is scored with half marks, so the fitted count is the doubled
# scale out of 80; everything the report shows is divided back to whole marks.
DISPLAY_RESCALE = {"EI": (0.5, 40, "whole marks (half marks doubled for fitting)")}

GB_WORD_READING = {"level": "lrp-rli-gbl-012", "gain": "lrp-rli-gbg-012"}


# --- artefact access ------------------------------------------------------


class Artefacts:
    """Reader over one sampling configuration's fitted output."""

    def __init__(self, config: str) -> None:
        self.config = config
        self.stat_root = paths.stat_models_dir()
        self.gb_root = paths.gb_models_dir()
        self.missing: list[str] = []

    def stat_dir(self, model_id: str) -> Path:
        return self.stat_root / f"{model_id}-{self.config}"

    def table(self, model_id: str, name: str) -> pd.DataFrame | None:
        path = self.stat_dir(model_id) / f"{name}.csv"
        if not path.is_file():
            self.missing.append(f"{model_id}:{name}.csv")
            return None
        return pd.read_csv(path)

    def config_json(self, model_id: str) -> dict[str, Any] | None:
        path = self.stat_dir(model_id) / "config.json"
        if not path.is_file():
            self.missing.append(f"{model_id}:config.json")
            return None
        return json.loads(path.read_text())

    def release(self, model_id: str) -> dict[str, Any] | None:
        path = self.stat_dir(model_id) / "release_decision.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def gb_metrics(self, model_id: str) -> dict[str, Any] | None:
        path = self.gb_root / model_id / "metrics.json"
        if not path.is_file():
            self.missing.append(f"{model_id}:metrics.json")
            return None
        return json.loads(path.read_text())

    def gb_table(self, model_id: str, name: str) -> pd.DataFrame | None:
        path = self.gb_root / model_id / f"{name}.csv"
        if not path.is_file():
            self.missing.append(f"{model_id}:{name}.csv")
            return None
        return pd.read_csv(path)


def _row(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """First row of a one-row summary table as a plain dict."""
    if df is None or df.empty:
        return None
    return {k: (None if pd.isna(v) else v) for k, v in df.iloc[0].to_dict().items()}


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


# --- section builders -----------------------------------------------------


def build_itt_suite(art: Artefacts) -> list[dict[str, Any]]:
    """The headline randomised timepoint-2 contrast for every outcome."""
    rows: list[dict[str, Any]] = []
    for symbol in OUTCOME_ORDER:
        model_id = ITT_PRIMARY.get(symbol)
        if model_id is None:
            continue
        rope = _row(art.table(model_id, "rope_summary"))
        cfg = art.config_json(model_id)
        if rope is None or cfg is None:
            continue
        measure = MEASURES.get(symbol)
        estimand = str(cfg.get("estimand_type") or "")
        floor_rule = "risk_difference" in estimand

        # A floor-rule fit estimates a difference in the chance of moving off the
        # floor, so its "items" columns are a probability difference; every other
        # fit is already on the test's item scale. APT information is rescaled
        # from its doubled half-mark fitting scale back to whole marks.
        scale, n_trials_display, scale_note = DISPLAY_RESCALE.get(
            symbol, (1.0, measure.n_trials if measure else None, None)
        )
        if floor_rule:
            scale, unit = 100.0, "percentage points"
            n_trials_display = measure.n_trials if measure else None
        else:
            unit = "items"

        def _scaled(key: str) -> float | None:
            value = _f(rope.get(key))
            return None if value is None else value * scale

        label = measure.label if measure else symbol
        if symbol in DISPLAY_RESCALE:
            label = label.replace(", half marks", "")

        rows.append(
            {
                "symbol": symbol,
                "label": label,
                "n_trials": n_trials_display,
                "scale_note": scale_note,
                "model_id": model_id,
                "n_children": cfg.get("n_obs"),
                "floor_rule": floor_rule,
                "unit": unit,
                "median": _scaled("items_median"),
                "lo": _scaled("items_lo"),
                "hi": _scaled("items_hi"),
                "lo50": _scaled("items_lo50"),
                "hi50": _scaled("items_hi50"),
                "prob_direction": _f(rope.get("pd")),
                "favoured_direction": rope.get("favoured_direction"),
                "evidence_label": rope.get("favoured_direction_label"),
                "delta": _scaled("delta_items"),
                "prob_benefit_ge_delta": _f(rope.get("prob_benefit_ge_delta")),
                "prob_in_rope": _f(rope.get("prob_in_rope")),
                "release": (art.release(model_id) or {}).get("status"),
            }
        )
    return rows


def build_triangulation(art: Artefacts) -> list[dict[str, Any]]:
    """The same randomised contrast as four model families estimate it."""
    rows: list[dict[str, Any]] = []
    for symbol in OUTCOME_ORDER:
        entry: dict[str, Any] = {"symbol": symbol}
        measure = MEASURES.get(symbol)
        entry["label"] = measure.label if measure else symbol

        itt = _row(art.table(ITT_PRIMARY[symbol], "rope_summary")) if symbol in ITT_PRIMARY else None
        if itt:
            entry["itt"] = {
                "model_id": ITT_PRIMARY[symbol],
                "median": _f(itt.get("items_median")),
                "lo": _f(itt.get("items_lo")),
                "hi": _f(itt.get("items_hi")),
                "prob": _f(itt.get("pd")),
            }

        if symbol in DID_PRIMARY:
            did = _row(art.table(DID_PRIMARY[symbol], "did_summary"))
            if did:
                entry["did"] = {
                    "model_id": DID_PRIMARY[symbol],
                    "median": _f(did.get("tau_t2_items_median")),
                    "lo": _f(did.get("tau_t2_items_lo")),
                    "hi": _f(did.get("tau_t2_items_hi")),
                    "prob": _f(did.get("prob_tau_t2_pos")),
                }

        if symbol in GF_PRIMARY:
            gf = _row(art.table(GF_PRIMARY[symbol], "treatment_marginal"))
            if gf:
                entry["gain_factors"] = {
                    "model_id": GF_PRIMARY[symbol],
                    "median": _f(gf.get("trt_items_median")),
                    "lo": _f(gf.get("trt_items_lo")),
                    "hi": _f(gf.get("trt_items_hi")),
                    "prob": _f(gf.get("prob_trt_pos")),
                }

        if symbol in LF_PRIMARY:
            lf = art.table(LF_PRIMARY[symbol], "factor_summary")
            if lf is not None and "term" in lf.columns:
                focal = lf[lf["term"].astype(str).str.startswith("d_grp_time[t2]")]
                if not focal.empty:
                    r = focal.iloc[0]
                    entry["level_factors"] = {
                        "model_id": LF_PRIMARY[symbol],
                        "logit_median": _f(r.get("median")),
                        "logit_lo": _f(r.get("lo")),
                        "logit_hi": _f(r.get("hi")),
                        "prob": _f(r.get("prob_positive")),
                        "term": str(r.get("term")),
                    }
        rows.append(entry)
    return rows


def build_word_reading(art: Artefacts) -> dict[str, Any]:
    """The headline word-reading result with its robustness and assumption envelope."""
    out: dict[str, Any] = {}
    head = _row(art.table("lrp-rli-itt-010", "rope_summary"))
    cfg = art.config_json("lrp-rli-itt-010")
    if head and cfg:
        out["headline"] = {
            "model_id": "lrp-rli-itt-010",
            "n_children": cfg.get("n_obs"),
            "median": _f(head.get("items_median")),
            "lo": _f(head.get("items_lo")),
            "hi": _f(head.get("items_hi")),
            "lo50": _f(head.get("items_lo50")),
            "hi50": _f(head.get("items_hi50")),
            "prob_direction": _f(head.get("pd")),
            "evidence_label": head.get("favoured_direction_label"),
            "delta": _f(head.get("delta_items")),
            "prob_benefit_ge_delta": _f(head.get("prob_benefit_ge_delta")),
            "prob_in_rope": _f(head.get("prob_in_rope")),
            "analysis_set": cfg.get("analysis_set_by_arm"),
        }

    robustness = []
    for model_id, label in WORD_READING_ROBUSTNESS:
        rope = _row(art.table(model_id, "rope_summary"))
        cfg_r = art.config_json(model_id)
        if rope and cfg_r:
            robustness.append(
                {
                    "model_id": model_id,
                    "label": label,
                    "n_children": cfg_r.get("n_obs"),
                    "median": _f(rope.get("items_median")),
                    "lo": _f(rope.get("items_lo")),
                    "hi": _f(rope.get("items_hi")),
                    "prob_direction": _f(rope.get("pd")),
                }
            )
    out["robustness"] = robustness

    sens = art.table("lrp-rli-itt-010", "itt_missingness_sensitivity")
    if sens is not None:
        named = {
            "screening_model_observed_profiles": "Same 53 children, screening reading and age instead of own baseline",
            "mar_all_57": "All 57 randomised, assuming the missing scores are missing at random",
            "jump_to_reference_intervention_nonstarter": "Both randomised arms completed, non-starter given the control pattern",
        }
        scenarios = []
        for row in json.loads(sens.to_json(orient="records")):
            if row.get("scenario") in named:
                scenarios.append(
                    {
                        "scenario": row["scenario"],
                        "label": named[row["scenario"]],
                        "estimand_class": row.get("estimand_class"),
                        "target_population": row.get("target_population"),
                        "median": _f(row.get("effect_items_median")),
                        "lo": _f(row.get("effect_items_lo89")),
                        "hi": _f(row.get("effect_items_hi89")),
                        "prob_direction": _f(row.get("prob_effect_positive")),
                    }
                )
        grid = sens[sens["scenario_class"] == "arm_specific_delta_grid"]
        out["missingness"] = scenarios
        if not grid.empty:
            out["missingness_delta_grid"] = {
                "n_cells": int(len(grid)),
                "median_min": _f(grid["effect_items_median"].min()),
                "median_max": _f(grid["effect_items_median"].max()),
                "prob_min": _f(grid["prob_effect_positive"].min()),
                "prob_max": _f(grid["prob_effect_positive"].max()),
            }
    bounds = _row(art.table("lrp-rli-itt-010", "attrition_bounds"))
    if bounds is not None:
        out["attrition_bounds"] = {
            "observed_items_difference": _f(bounds.get("observed_items_difference")),
            "worst_case_items_lower": _f(bounds.get("worst_case_items_lower")),
            "worst_case_items_upper": _f(bounds.get("worst_case_items_upper")),
            "missing_intervention_n": bounds.get("missing_intervention_n"),
            "missing_control_n": bounds.get("missing_control_n"),
            "interpretation": bounds.get("interpretation"),
        }
    return out


def build_mediation(art: Artefacts) -> list[dict[str, Any]]:
    """How the word-reading contrast divides between routes."""
    rows: list[dict[str, Any]] = []
    for model_id, label in MEDIATION_ROUTES:
        table = art.table(model_id, "mediation_summary")
        if table is None:
            continue
        rows.append({"model_id": model_id, "route": label, "rows": json.loads(table.to_json(orient="records"))})
    return rows


def build_mechanism(art: Artefacts) -> dict[str, Any]:
    """Skill-to-skill associations, per SD and on the outcome's own item scale."""
    out: dict[str, Any] = {}

    # The cross-model comparison is the only place the R/E/L slopes are put on one
    # scale (per SD of the predictor's logit), so read it rather than re-deriving.
    forest = paths.stat_dir() / "comparison" / "mechanism_forest.csv"
    if forest.is_file():
        out["per_sd"] = json.loads(pd.read_csv(forest).to_json(orient="records"))

    rows: list[dict[str, Any]] = []
    for model_id, label in MECHANISM_SLOPES:
        summary = _row(art.table(model_id, "mechanism_summary"))
        cfg = art.config_json(model_id)
        if summary is None or cfg is None:
            continue
        rows.append(
            {
                "model_id": model_id,
                "label": label,
                "outcome_symbol": cfg.get("outcome_symbol"),
                "mechanism_symbol": cfg.get("mechanism_symbol"),
                "n_rows": cfg.get("n_obs"),
                "exposure_low": _f(summary.get("exposure_low")),
                "exposure_high": _f(summary.get("exposure_high")),
                "exposure_unit": summary.get("exposure_unit"),
                "items_median": _f(summary.get("items_median")),
                "items_lo": _f(summary.get("items_lo")),
                "items_hi": _f(summary.get("items_hi")),
                "prob_pos": _f(summary.get("prob_pos")),
            }
        )
    out["items_scale"] = rows
    return out


def build_gb(art: Artefacts) -> dict[str, Any]:
    """Step-1 gradient-boosting performance and predictor rankings."""
    performance: list[dict[str, Any]] = []
    for model_id in sorted(p.name for p in art.gb_root.glob("lrp-rli-gb*")):
        metrics = art.gb_metrics(model_id)
        if metrics is None:
            continue
        performance.append(
            {
                "model_id": model_id,
                "target": metrics.get("target_var"),
                "kind": "gain" if "-gbg-" in model_id else "level",
                "n_observations": metrics.get("n_observations"),
                "pooled_r2": _f(metrics.get("cv_pooled_r2")),
                "pooled_mae": _f(metrics.get("cv_pooled_mae")),
            }
        )

    rankings: dict[str, Any] = {}
    for kind, model_id in GB_WORD_READING.items():
        ranking = art.gb_table(model_id, "predictor_ranking")
        if ranking is None:
            continue
        top = ranking.sort_values("cluster_rank").head(10)
        rankings[kind] = {
            "model_id": model_id,
            "rows": json.loads(top.to_json(orient="records")),
        }
    return {"performance": performance, "word_reading_rankings": rankings}


def build_horseshoe(art: Artefacts) -> dict[str, Any]:
    """The Bayesian many-predictor ranking that cross-checks the GB one."""
    out: dict[str, Any] = {}
    for model_id in ("lrp-rli-hs-001", "lrp-rli-hs-002"):
        ranking = art.table(model_id, "predictor_ranking")
        cfg = art.config_json(model_id)
        if ranking is None or cfg is None:
            continue
        out[model_id] = {
            "title": cfg.get("title"),
            "outcome_symbol": cfg.get("outcome_symbol"),
            "n_rows": cfg.get("n_obs"),
            "rows": json.loads(ranking.head(10).to_json(orient="records")),
        }
    return out


def build_pooled_levels(art: Artefacts) -> list[dict[str, Any]]:
    """Between-child versus within-child level associations."""
    rows: list[dict[str, Any]] = []
    for model_id in sorted(p.name.replace(f"-{art.config}", "") for p in art.stat_root.glob(f"lrp-rli-pl-*-{art.config}")):
        cfg = art.config_json(model_id)
        table = art.table(model_id, "pooled_levels_summary")
        if cfg is None:
            continue
        entry: dict[str, Any] = {
            "model_id": model_id,
            "title": cfg.get("title"),
            "outcome_symbol": cfg.get("outcome_symbol"),
            "mechanism_symbol": cfg.get("mechanism_symbol"),
            "n_rows": cfg.get("n_obs"),
        }
        if table is not None:
            entry["rows"] = json.loads(table.to_json(orient="records"))
        rows.append(entry)
    return rows


def build_release_inventory(art: Artefacts) -> dict[str, Any]:
    """What the publication gate decided across every fit at this config."""
    statuses: dict[str, int] = {}
    withheld: list[dict[str, str]] = []
    divergences = 0
    gate_failures: list[str] = []
    total = 0
    for directory in sorted(art.stat_root.glob(f"*-{art.config}")):
        decision_path = directory / "release_decision.json"
        if not decision_path.is_file():
            continue
        total += 1
        decision = json.loads(decision_path.read_text())
        status = str(decision.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        model_id = directory.name[: -len(f"-{art.config}")]
        if decision.get("publishable") is not True:
            withheld.append({"model_id": model_id, "status": status, "reason": str(decision.get("reason") or "")})
        diagnostics_path = directory / "diagnostics_summary.json"
        if diagnostics_path.is_file():
            diagnostics = json.loads(diagnostics_path.read_text())
            n_div = diagnostics.get("n_divergent", diagnostics.get("divergences"))
            if isinstance(n_div, (int, float)) and n_div > 0:
                divergences += 1
            passed = diagnostics.get("passed", diagnostics.get("gate_passed"))
            if passed is False:
                gate_failures.append(model_id)
    return {
        "n_fits": total,
        "statuses": statuses,
        "withheld": withheld,
        "n_fits_with_divergences": divergences,
        "gate_failures": gate_failures,
    }


# --- figures --------------------------------------------------------------


def _prob_text(prob: float | None) -> str:
    """Format a direction probability without ever rounding it up to certainty."""
    if prob is None:
        return "-"
    if prob >= 0.9995:
        return ">0.999"
    return f"{prob:.3f}" if prob >= 0.99 else f"{prob:.2f}"


def _interval_plot(
    ax: plt.Axes,
    labels: list[str],
    medians: list[float],
    lo: list[float],
    hi: list[float],
    lo50: list[float] | None = None,
    hi50: list[float] | None = None,
    colours: list[str] | None = None,
) -> None:
    """Horizontal median + inner-50% + outer-89% interval rows, top row first."""
    positions = np.arange(len(labels))[::-1]
    for index, position in enumerate(positions):
        colour = colours[index] if colours else "#1f4e79"
        ax.plot([lo[index], hi[index]], [position, position], color=colour, linewidth=1.6, solid_capstyle="round")
        if lo50 is not None and hi50 is not None:
            ax.plot([lo50[index], hi50[index]], [position, position], color=colour, linewidth=4.5, solid_capstyle="round", alpha=0.55)
        ax.plot([medians[index]], [position], marker="o", markersize=6, color=colour, zorder=3)
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def figure_effects_forest(results: dict[str, Any], out_dir: Path) -> None:
    rows = [r for r in results["itt_suite"] if not r["floor_rule"] and r["median"] is not None]
    if not rows:
        return
    labels = [f"{r['label'].split(' (')[0]} (of {r['n_trials']})" for r in rows]
    colours = ["#1f4e79" if (r["prob_direction"] or 0) >= 0.97 else "#4a7fb5" if (r["prob_direction"] or 0) >= 0.91 else "#9db8d2" for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 0.46 * len(rows) + 1.9))
    _interval_plot(
        ax,
        labels,
        [r["median"] for r in rows],
        [r["lo"] for r in rows],
        [r["hi"] for r in rows],
        [r["lo50"] for r in rows],
        [r["hi50"] for r in rows],
        colours,
    )
    for index, r in enumerate(rows):
        ax.text(
            ax.get_xlim()[1],
            (len(rows) - 1 - index),
            f"  P={_prob_text(r['prob_direction'])}",
            va="center",
            fontsize=8,
            color="#333333",
        )
    ax.set_xlabel("Difference in test items after 20 weeks (immediate minus waiting), each test on its own scale")
    ax.set_title("What the intervention changed, by outcome")
    save_styled_figure(str(out_dir), "effects_forest", fig=fig, data=pd.DataFrame(rows))


def figure_word_reading_envelope(results: dict[str, Any], out_dir: Path) -> None:
    word = results.get("word_reading", {})
    head = word.get("headline")
    if not head:
        return
    entries = [
        (f"Headline: the {head['n_children']} children with a timepoint-2 score", head["median"], head["lo"], head["hi"])
    ]
    for row in word.get("missingness", []):
        label, median, lo, hi = row.get("label"), row.get("median"), row.get("lo"), row.get("hi")
        if label and median is not None and lo is not None and hi is not None:
            entries.append((label, median, lo, hi))
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(entries) + 2.2))
    _interval_plot(
        ax,
        [e[0] for e in entries],
        [e[1] for e in entries],
        [e[2] for e in entries],
        [e[3] for e in entries],
    )
    bounds = word.get("attrition_bounds") or {}
    lo_bound = bounds.get("worst_case_items_lower")
    hi_bound = bounds.get("worst_case_items_upper")
    if lo_bound is not None and hi_bound is not None:
        ax.axvspan(lo_bound, hi_bound, color="#d9534f", alpha=0.07, zorder=0)
        ax.text(
            lo_bound,
            len(entries) - 0.35,
            f"  if the {int(bounds.get('missing_intervention_n') or 0) + int(bounds.get('missing_control_n') or 0)} "
            f"missing children took the most extreme possible scores: {lo_bound:+.1f} to {hi_bound:+.1f} words",
            fontsize=8,
            color="#a33",
            va="bottom",
        )
    ax.set_xlabel("Word-reading difference in words (of 79), immediate minus waiting")
    ax.set_title("Word reading: the headline estimate and what the missing scores could do to it")
    save_styled_figure(str(out_dir), "word_reading_envelope", fig=fig, data=pd.DataFrame(entries, columns=["analysis", "median", "lo", "hi"]))


def figure_trajectories(art: Artefacts, out_dir: Path, model_id: str, name: str, outcome: str) -> pd.DataFrame | None:
    traj = art.table(model_id, "group_trajectory")
    if traj is None or traj.empty:
        return None
    waves = sorted(str(t) for t in traj["timepoint"].unique())
    title = f"{outcome}: fitted group means at {', '.join(waves)}"
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colours = {"immediate intervention": "#1f4e79", "wait-list control": "#c8791a"}
    for arm_label, group in traj.groupby("arm_label"):
        group = group.sort_values("wave")
        x = np.arange(len(group))
        colour = colours.get(str(arm_label), "#555555")
        ax.plot(x, group["predicted_median"], marker="o", color=colour, label=str(arm_label))
        ax.fill_between(x, group["predicted_lo"], group["predicted_hi"], color=colour, alpha=0.15)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in group["timepoint"]])
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=1.2)
    ax.text(1.02, ax.get_ylim()[1], " waiting group starts here", fontsize=8, va="top", color="#666666")
    ax.set_xlabel("Assessment timepoint")
    ax.set_ylabel("Score (items)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    save_styled_figure(str(out_dir), name, fig=fig, data=traj)
    return traj


def figure_mechanism(results: dict[str, Any], out_dir: Path) -> None:
    """Per-SD slopes of the comparable R/E/L trio, the only common scale they share."""
    rows = (results.get("mechanism") or {}).get("per_sd") or []
    rows = [r for r in rows if r.get("beta_mech_median") is not None]
    if not rows:
        return
    names = {"L": "Letter-sound knowledge", "E": "Expressive vocabulary", "R": "Receptive vocabulary"}
    rows = sorted(rows, key=lambda r: -float(r["beta_mech_median"]))
    fig, ax = plt.subplots(figsize=(8.2, 0.7 * len(rows) + 2.4))
    _interval_plot(
        ax,
        [names.get(str(r.get("mechanism") or r.get("symbol")), str(r.get("model"))) for r in rows],
        [float(r["beta_mech_median"]) for r in rows],
        [float(r["beta_mech_lo"]) for r in rows],
        [float(r["beta_mech_hi"]) for r in rows],
        [float(r["beta_mech_lo50"]) for r in rows] if "beta_mech_lo50" in rows[0] else None,
        [float(r["beta_mech_hi50"]) for r in rows] if "beta_mech_hi50" in rows[0] else None,
    )
    ax.set_xlabel("Change in later word reading (log-odds) per standard deviation of the earlier skill")
    ax.set_title("Which earlier skills track later word reading (associations, not levers)")
    save_styled_figure(str(out_dir), "mechanism_forest", fig=fig, data=pd.DataFrame(rows))


def figure_gb_predictability(results: dict[str, Any], out_dir: Path) -> None:
    performance = pd.DataFrame(results.get("gb", {}).get("performance") or [])
    if performance.empty:
        return
    performance = performance.dropna(subset=["pooled_r2"])
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for kind, colour in (("level", "#1f4e79"), ("gain", "#c8791a")):
        subset = performance[performance["kind"] == kind]["pooled_r2"]
        if subset.empty:
            continue
        ax.scatter(
            np.random.default_rng(7).normal(0 if kind == "level" else 1, 0.06, len(subset)),
            subset,
            s=26,
            alpha=0.75,
            color=colour,
            label=f"{kind} models (n={len(subset)})",
        )
        ax.plot([(0 if kind == "level" else 1) - 0.22, (0 if kind == "level" else 1) + 0.22], [subset.median()] * 2, color=colour, linewidth=2.4)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Current level\n(where a child is now)", "Change over a period\n(how fast they moved)"])
    ax.set_ylabel("Out-of-sample variance explained (pooled $R^2$)")
    ax.set_title("What the exploratory models can and cannot predict")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    save_styled_figure(str(out_dir), "gb_predictability", fig=fig, data=performance)


def figure_gb_ranking(results: dict[str, Any], out_dir: Path) -> None:
    rankings = results.get("gb", {}).get("word_reading_rankings", {})
    level = rankings.get("level")
    if not level:
        return
    frame = pd.DataFrame(level["rows"])
    frame = frame[frame["mean_abs_shap"] > 0].head(8).iloc[::-1]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(frame) + 2.0))
    ax.barh(frame["member"], frame["mean_abs_shap"], color="#1f4e79", alpha=0.85)
    ax.set_xlabel("Mean absolute SHAP value (words of 79)")
    ax.set_title("Which measures the exploratory model leans on for word-reading level")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    save_styled_figure(str(out_dir), "gb_ranking_word_reading_level", fig=fig, data=frame)


# --- entry point ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="reporting", help="Sampling configuration to read")
    parser.add_argument("--output-dir", default=None, help="Override the output root")
    args = parser.parse_args(argv)

    setup.init_script()
    paths.set_output_root(args.output_dir)
    art = Artefacts(args.config)
    out_dir = paths.output_root() / "summary_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "meta": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "config": args.config,
            "output_root": str(paths.output_root()),
        },
        "itt_suite": build_itt_suite(art),
        "triangulation": build_triangulation(art),
        "word_reading": build_word_reading(art),
        "mediation": build_mediation(art),
        "mechanism": build_mechanism(art),
        "pooled_levels": build_pooled_levels(art),
        "gb": build_gb(art),
        "horseshoe": build_horseshoe(art),
        "release": build_release_inventory(art),
    }

    figure_effects_forest(results, out_dir)
    figure_word_reading_envelope(results, out_dir)
    figure_trajectories(art, out_dir, "lrp-rli-did-001", "trajectory_word_reading", "Word reading")
    figure_trajectories(art, out_dir, "lrp-rli-did-002", "trajectory_letter_sounds", "Letter-sound knowledge")
    figure_mechanism(results, out_dir)
    figure_gb_predictability(results, out_dir)
    figure_gb_ranking(results, out_dir)

    results["meta"]["missing_artefacts"] = sorted(set(art.missing))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_dir / 'results.json'}")
    if art.missing:
        print(f"Missing artefacts ({len(set(art.missing))}): {', '.join(sorted(set(art.missing))[:12])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
