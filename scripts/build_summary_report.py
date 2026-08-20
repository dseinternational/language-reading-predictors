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
    """The same randomised contrast as the treatment families estimate it.

    Prefers the cross-model comparison's own ``triangulation_consistency.csv``,
    which puts each family on its canonical items-scale average marginal effect
    and records whether the designs agree — quantities the report should not
    re-derive. Falls back to reading each family's summary directly.
    """
    canonical = paths.stat_dir() / "comparison" / "triangulation_consistency.csv"
    if canonical.is_file():
        table = pd.read_csv(canonical)
        rows_out: list[dict[str, Any]] = []
        for record in json.loads(table.to_json(orient="records")):
            symbol = str(record.get("outcome"))
            measure = MEASURES.get(symbol)
            entry: dict[str, Any] = {
                "symbol": symbol,
                "label": measure.label if measure else symbol,
                "source": "triangulation_consistency.csv",
                "direction_agree": bool(record.get("direction_agree")),
                "intervals_overlap": bool(record.get("intervals_overlap")),
                "consistent": bool(record.get("consistent")),
                "n_designs": record.get("n_designs"),
            }
            for prefix, key in (("itt", "itt"), ("did", "did"), ("gf", "gain_factors")):
                entry[key] = {
                    "model_id": record.get(f"{prefix}_source"),
                    "median": _f(record.get(f"{prefix}_items_median")),
                    "lo": _f(record.get(f"{prefix}_items_lo")),
                    "hi": _f(record.get(f"{prefix}_items_hi")),
                    "prob": _f(record.get(f"{prefix}_prob_pos")),
                    "estimand": record.get(f"{prefix}_estimand"),
                }
            rows_out.append(entry)
        return rows_out

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


def trace_summary(art: Artefacts, model_id: str, parameters: list[str]) -> dict[str, dict[str, Any]]:
    """House-standard posterior summary for named parameters, read from the trace.

    The per-fit ``diagnostics.csv`` carries a posterior *mean* and an 89% interval
    but neither a median nor a direction probability, and this project reports the
    median with a direction probability, so read the draws.
    """
    path = art.stat_dir(model_id) / "trace.nc"
    if not path.is_file():
        art.missing.append(f"{model_id}:trace.nc")
        return {}
    import arviz as az

    trace = az.from_netcdf(path)
    out: dict[str, dict[str, Any]] = {}
    for name in parameters:
        if name not in trace.posterior:
            art.missing.append(f"{model_id}:{name}")
            continue
        draws = np.asarray(trace.posterior[name].values).reshape(-1)
        out[name] = {
            "median": float(np.median(draws)),
            "lo50": float(np.quantile(draws, 0.25)),
            "hi50": float(np.quantile(draws, 0.75)),
            "lo": float(np.quantile(draws, 0.055)),
            "hi": float(np.quantile(draws, 0.945)),
            "prob_pos": float(np.mean(draws > 0)),
        }
    return out


def build_mediation(art: Artefacts) -> list[dict[str, Any]]:
    """How the word-reading contrast divides between routes (g-formula)."""
    rows: list[dict[str, Any]] = []
    for model_id, label in MEDIATION_ROUTES:
        table = art.table(model_id, "mediation_summary")
        if table is None or "quantity" not in table.columns:
            continue
        indexed = table.set_index("quantity")
        entry: dict[str, Any] = {"model_id": model_id, "route": label}
        for quantity, key in (("total", "total"), ("NIE", "through_route"), ("NDE", "not_through_route")):
            if quantity not in indexed.index:
                continue
            row = indexed.loc[quantity]
            entry[key] = {
                "median": _f(row.get("words_median")),
                "lo": _f(row.get("words_lo")),
                "hi": _f(row.get("words_hi")),
                "prob_pos": _f(row.get("prob_pos")),
            }
        if "proportion_mediated" in indexed.index:
            entry["proportion_mediated_median"] = _f(indexed.loc["proportion_mediated"].get("prob_median"))
        rows.append(entry)
    return rows


def build_mechanism(art: Artefacts) -> dict[str, Any]:
    """Skill-to-skill associations, per SD and on the outcome's own item scale."""
    out: dict[str, Any] = {}

    # The cross-model comparison is the only place the R/E/L slopes are put on one
    # scale (per SD of the predictor's logit), so read it rather than re-deriving:
    # the curve model has no single slope, and the comparison converts its average
    # gradient onto the same per-SD scale as the two linear models.
    forest = paths.stat_dir() / "comparison" / "mechanism_forest.csv"
    if forest.is_file():
        table = pd.read_csv(forest)
        per_sd = []
        for record in json.loads(table.to_json(orient="records")):
            name = str(record.get("model", ""))
            symbol = name.split("(")[-1].split("->")[0].strip() if "(" in name else None
            per_sd.append(
                {
                    "model_id": name.split(" ")[0],
                    "mechanism_symbol": symbol,
                    "shape": record.get("mechanism_shape"),
                    "slope_mean_per_sd": _f(record.get("slope_mean")),
                    "lo": _f(record.get("slope_lo")),
                    "hi": _f(record.get("slope_hi")),
                }
            )
        out["per_sd"] = per_sd

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


PREDICTOR_LABELS = {
    "L": "Letter-sound knowledge",
    "E": "Expressive vocabulary",
    "R": "Receptive vocabulary",
    "erbto": "Phonological memory (word/nonword repetition)",
    "deapp_c": "Speech clarity",
    "hs": "Hearing-difficulty flag",
}


def build_pooled_levels(art: Artefacts) -> list[dict[str, Any]]:
    """Between-child versus within-child level associations across the four waves."""
    rows: list[dict[str, Any]] = []
    for directory in sorted(art.stat_root.glob(f"lrp-rli-pl-*-{art.config}")):
        model_id = directory.name[: -len(f"-{art.config}")]
        cfg = art.config_json(model_id)
        table = art.table(model_id, "pooled_levels_summary")
        if cfg is None or table is None or "term" not in table.columns:
            continue
        indexed = table.set_index("term")
        entry: dict[str, Any] = {
            "model_id": model_id,
            "title": cfg.get("title"),
            "outcome_symbol": cfg.get("outcome_symbol"),
            "mechanism_symbol": cfg.get("mechanism_symbol"),
            "predictor_label": PREDICTOR_LABELS.get(str(cfg.get("mechanism_symbol")), str(cfg.get("mechanism_symbol"))),
            "n_rows": cfg.get("n_obs"),
        }
        for term, key in (("beta_between", "between"), ("beta_within", "within")):
            if term in indexed.index:
                row = indexed.loc[term]
                entry[key] = {
                    "median": _f(row.get("median")),
                    "lo": _f(row.get("lo")),
                    "hi": _f(row.get("hi")),
                    "prob_pos": _f(row.get("prob_positive")),
                }
        items_term = "beta_between (items per +1 SD)"
        if items_term in indexed.index:
            row = indexed.loc[items_term]
            entry["between_items_per_sd"] = {
                "median": _f(row.get("median")),
                "lo": _f(row.get("lo")),
                "hi": _f(row.get("hi")),
            }
        rows.append(entry)
    return rows


# Each family names its coefficients differently, so the parameter names are
# given per model rather than assumed. ``hearing`` and ``speech`` name the role
# the coefficient plays, which is what the report reads.
HEARING_SPEECH_VIEWS = [
    (
        "lrp-rli-mech-058",
        "Gain over one period, given the score at the start of it",
        {"hearing": "gamma_hs", "speech": "gamma_deapp_c"},
    ),
    (
        "lrp-rli-lcsm-082",
        "Latent change in reading and blending",
        {"hearing": "b_hs", "speech": "b_deapp_c"},
    ),
    (
        "lrp-rli-mm-002",
        "Latent code-factor gain",
        {"hearing": "beta_hs", "speech": "beta_deapp_c"},
    ),
    (
        "lrp-rli-pl-001",
        "Level pooled across the four assessments",
        {"hearing": "gamma_hs", "speech": "gamma_deapp_c"},
    ),
]


def build_hearing_speech(art: Artefacts) -> dict[str, Any]:
    """The hearing flag and the speech-clarity score as they appear across fits.

    Every one of these coefficients was included to clean up some *other*
    estimate, so the report states them as adjusted associations and says so.
    """
    out: dict[str, Any] = {"views": [], "whole_study_gain": {}}

    words = art.table("lrp-rli-adj-065", "predicted_gain_words")
    assoc = art.table("lrp-rli-adj-065", "predictor_associations")
    if words is not None and assoc is not None and "predictor" in words.columns:
        merged = words.merge(assoc[["predictor", "adj_prob_pos"]], on="predictor", how="left")
        for predictor in ("hs", "deapp_c", "erbto", "L"):
            hit = merged[merged["predictor"] == predictor]
            if hit.empty:
                continue
            row = hit.iloc[0]
            out["whole_study_gain"][predictor] = {
                "model_id": "lrp-rli-adj-065",
                "label": PREDICTOR_LABELS.get(predictor, str(row.get("label"))),
                "words_median": _f(row.get("delta_words_median")),
                "words_lo": _f(row.get("delta_words_lo")),
                "words_hi": _f(row.get("delta_words_hi")),
                "prob_pos": _f(row.get("prob_pos")),
            }

    for model_id, label, parameters in HEARING_SPEECH_VIEWS:
        summary = trace_summary(art, model_id, list(parameters.values()))
        if summary:
            out["views"].append(
                {
                    "model_id": model_id,
                    "label": label,
                    "parameters": {
                        role: {**summary[name], "parameter": name}
                        for role, name in parameters.items()
                        if name in summary
                    },
                }
            )

    out["randomised_window_by_subgroup"] = randomised_window_subgroups()
    return out


def build_design(art: Artefacts) -> dict[str, Any]:
    """The trial's shape as the analysis files record it.

    Taken from the data rather than from the published design description: the
    two differ after timepoint 3, where both arms received further sessions.
    """
    frame = pd.read_csv(REPO_ROOT / "data" / "rli_data_long.csv")
    arms = {1: "immediate", 2: "waiting"}
    waves: list[dict[str, Any]] = []
    for time in sorted(frame["time"].unique()):
        wave = frame[frame["time"] == time]
        entry: dict[str, Any] = {
            "timepoint": f"t{int(time)}",
            "age_months_all": _f(wave["age"].mean()),
        }
        for code, name in arms.items():
            arm = wave[wave["group"] == code]
            entry[f"n_{name}"] = int(arm["subject_id"].nunique())
            entry[f"age_months_{name}"] = _f(arm["age"].mean())
            entry[f"cumulative_sessions_{name}"] = _f(arm["attend_cumul"].mean())
        waves.append(entry)

    baseline = frame[frame["time"] == 1]
    return {
        "waves": waves,
        "n_analysed": int(frame["subject_id"].nunique()),
        "age_months_min": _f(baseline["age"].min()),
        "age_months_max": _f(baseline["age"].max()),
        "age_months_mean": _f(baseline["age"].mean()),
    }


def randomised_window_subgroups() -> list[dict[str, Any]]:
    """Descriptive t1-to-t2 word-reading gains by arm within baseline subgroups.

    Descriptive, not modelled: the trial cannot support a subgroup treatment
    estimate at this size, and the report says so. Computed here rather than
    quoted from a note so the numbers come from the data in this checkout.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        derive_hearing_composite,
    )

    frame = pd.read_csv(REPO_ROOT / "data" / "rli_data_long.csv")
    wide = frame.pivot_table(index="subject_id", columns="time", values="ewrswr")
    if 1 not in wide.columns or 2 not in wide.columns:
        return []
    gain = (wide[2] - wide[1]).rename("gain")

    baseline = frame[frame["time"] == 1].set_index("subject_id")
    try:
        hearing = derive_hearing_composite(baseline)
    except Exception:  # pragma: no cover - loader shape changes
        hearing = baseline.get("hearing_c")

    speech = baseline.get("deapp_c")
    rows: list[dict[str, Any]] = []
    splits: list[tuple[str, pd.Series]] = []
    if hearing is not None:
        series = pd.Series(hearing, index=baseline.index)
        splits.append(("Flagged for hearing difficulty", series == 1))
        splits.append(("Hearing recorded clear", series == 0))
    if speech is not None:
        median = speech.median()
        splits.append(("Speech clarity at or below the median", speech <= median))
        splits.append(("Speech clarity above the median", speech > median))

    for label, mask in splits:
        subset = baseline[mask.reindex(baseline.index, fill_value=False)]
        joined = subset.join(gain).dropna(subset=["gain"])
        if joined.empty:
            continue
        by_arm = joined.groupby("group")["gain"].agg(["count", "mean"])
        if 1 not in by_arm.index or 2 not in by_arm.index:
            continue
        rows.append(
            {
                "subgroup": label,
                "n_immediate": int(by_arm.loc[1, "count"]),
                "n_waiting": int(by_arm.loc[2, "count"]),
                "mean_gain_immediate": float(by_arm.loc[1, "mean"]),
                "mean_gain_waiting": float(by_arm.loc[2, "mean"]),
                "difference": float(by_arm.loc[1, "mean"] - by_arm.loc[2, "mean"]),
            }
        )
    return rows


def build_prior_dependence(art: Artefacts) -> dict[str, Any]:
    """How far a treatment estimate moves when its prior is widened.

    Attached by the family prior-sensitivity runners after the primary fit. Two
    of the treatment families need this evidence before their result may be
    released, and the report says what it establishes: direction, not size.
    """
    out: dict[str, Any] = {}
    for model_id, label in (("lrp-rli-did-001", "Arm-by-wave model of word reading"),):
        table = art.table(model_id, "tau_prior_sensitivity")
        if table is None or "tau_sigma" not in table.columns:
            continue
        grid = table.sort_values("tau_sigma")
        out[model_id] = {
            "label": label,
            "cells": [
                {
                    "prior_sd": _f(row.get("tau_sigma")),
                    "items": _f(row.get("items_mean")),
                    "lo": _f(row.get("items_lo")),
                    "hi": _f(row.get("items_hi")),
                    "prob_direction": _f(row.get("pd")),
                    "converged": bool(row.get("converged")),
                }
                for row in json.loads(grid.to_json(orient="records"))
            ],
        }
    return out


HORSESHOE_GB_PAIRS = [
    ("lrp-rli-hs-002", "lrp-rli-gbl-012", "Word reading", "level"),
    ("lrp-rli-hs-004", "lrp-rli-gbl-009", "Letter sounds", "level"),
    ("lrp-rli-hs-001", "lrp-rli-gbg-012", "Word reading", "change over a period"),
    ("lrp-rli-hs-003", "lrp-rli-gbg-009", "Letter sounds", "change over a period"),
]


def build_ranking_agreement(art: Artefacts) -> list[dict[str, Any]]:
    """Do the two step-1/step-2 ranking methods agree with each other?

    ``scripts/compare_horseshoe_vs_gb.py`` writes the aligned per-construct ranks;
    the rank correlation is recomputed here so the report's number comes from the
    stored comparison rather than from a console line someone transcribed.
    """
    from scipy.stats import spearmanr

    rows: list[dict[str, Any]] = []
    for hs_id, gb_id, outcome, kind in HORSESHOE_GB_PAIRS:
        path = paths.stat_dir() / "comparison" / f"horseshoe_vs_gb_{hs_id.replace('lrp-rli-', '')}.csv"
        if not path.is_file():
            art.missing.append(path.name)
            continue
        table = pd.read_csv(path)
        if {"p_abs_gt_delta", "gb_perm_imp"} - set(table.columns) or len(table) < 3:
            continue
        # Correlate the two importance scores directly, letting scipy apply the
        # average-rank tie correction, exactly as compare_horseshoe_vs_gb.py does.
        # Top-k overlap is deliberately not recomputed here: that tool takes it
        # from the *full* rankings, and this file holds only shared constructs,
        # so a value computed from it would disagree with the tool's own output.
        result = spearmanr(table["p_abs_gt_delta"], table["gb_perm_imp"])
        rows.append(
            {
                "horseshoe_model": hs_id,
                "gb_model": gb_id,
                "outcome": outcome,
                "quantity": kind,
                "n_shared_constructs": int(len(table)),
                "spearman_rho": float(result.statistic),
                "spearman_p": float(result.pvalue),
            }
        )
    return rows


def build_blending_link(art: Artefacts) -> list[dict[str, Any]]:
    """The mandatory phoneme-blending pair: ordinary logit and guessing-floor link.

    Neither result may be reported without the other, so the report reads both
    from the validated bundle rather than quoting the primary alone.
    """
    path = paths.stat_dir() / "blending_link_sensitivity" / "blending_link_sensitivity.csv"
    if not path.is_file():
        art.missing.append("blending_link_sensitivity.csv")
        return []
    table = pd.read_csv(path)
    table = table[table["config"] == art.config] if "config" in table.columns else table
    rows = []
    for row in json.loads(table.to_json(orient="records")):
        rows.append(
            {
                "model_id": row.get("model_id"),
                "link": row.get("score_mean_link"),
                "n": row.get("n"),
                "median": _f(row.get("effect_items_median")),
                "lo": _f(row.get("effect_items_lo")),
                "hi": _f(row.get("effect_items_hi")),
                "prob_pos": _f(row.get("prob_effect_positive")),
                "prob_meaningful": _f(row.get("prob_meaningful_benefit")),
                "elpd_difference": _f(row.get("guessing_floor_minus_logit_elpd")),
                "elpd_difference_se": _f(row.get("guessing_floor_minus_logit_elpd_se")),
            }
        )
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
    # The registry also holds models of a separate historical cohort (``rlm``),
    # which this report does not cover; every withheld fit is one of them, so the
    # counts must be able to say so.
    n_trial = sum(1 for directory in art.stat_root.glob(f"*-{art.config}") if "-rli-" in directory.name)
    return {
        "n_fits": total,
        "n_fits_trial": n_trial,
        "n_fits_other_cohort": total - n_trial,
        "statuses": statuses,
        "withheld": withheld,
        "n_withheld_other_cohort": sum(1 for row in withheld if "-rlm-" in row["model_id"]),
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
    fig, ax = plt.subplots(figsize=(9.0, 0.52 * len(entries) + 1.6))
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
        n_missing = int(bounds.get("missing_intervention_n") or 0) + int(bounds.get("missing_control_n") or 0)
        ax.axvspan(lo_bound, hi_bound, color="#d9534f", alpha=0.07, zorder=0)
        ax.text(
            lo_bound,
            -0.62,
            f"if the {n_missing} missing scores took the most extreme values possible: "
            f"{lo_bound:+.1f} to {hi_bound:+.1f} words",
            fontsize=8.5,
            color="#a33",
            va="center",
        )
        ax.set_ylim(-1.0, len(entries) - 0.4)
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
    rows = [r for r in rows if r.get("slope_mean_per_sd") is not None]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -float(r["slope_mean_per_sd"]))
    fig, ax = plt.subplots(figsize=(8.4, 0.8 * len(rows) + 2.4))
    _interval_plot(
        ax,
        [PREDICTOR_LABELS.get(str(r.get("mechanism_symbol")), str(r.get("model_id"))) for r in rows],
        [float(r["slope_mean_per_sd"]) for r in rows],
        [float(r["lo"]) for r in rows],
        [float(r["hi"]) for r in rows],
    )
    ax.set_xlabel("Association with later word reading (log-odds) per standard deviation of the earlier skill")
    ax.set_title("Which earlier skills track later word reading (associations, not levers)")
    save_styled_figure(str(out_dir), "mechanism_forest", fig=fig, data=pd.DataFrame(rows))


def figure_mediation(results: dict[str, Any], out_dir: Path) -> None:
    """The word-reading contrast split into the part through each route and the rest."""
    rows = [r for r in (results.get("mediation") or []) if r.get("through_route")]
    if not rows:
        return
    labels = [r["route"] for r in rows]
    through = [r["through_route"]["median"] for r in rows]
    direct = [r["not_through_route"]["median"] if r.get("not_through_route") else 0.0 for r in rows]
    positions = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(8.8, 0.62 * len(rows) + 2.0))
    ax.barh(positions, through, height=0.42, color="#1f4e79", label="through this route")
    ax.barh(positions, direct, height=0.42, left=through, color="#c8791a", alpha=0.85, label="not through this route")
    for index, position in enumerate(positions):
        entry = rows[index]["through_route"]
        ax.plot([entry["lo"], entry["hi"]], [position, position], color="#0d2b45", linewidth=1.4)
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Words of 79 attributed to each part of the 20-week difference")
    ax.set_title("Which route the word-reading benefit ran through")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    save_styled_figure(str(out_dir), "mediation_routes", fig=fig, data=pd.DataFrame(rows))


def figure_pooled_levels(results: dict[str, Any], out_dir: Path) -> None:
    """Between-child against within-child associations, side by side."""
    # pl-101 is a technical comparator (the same association without wave
    # intercepts), so it belongs on its model page rather than in this figure.
    rows = [
        r
        for r in (results.get("pooled_levels") or [])
        if r.get("between") and r.get("within") and not r["model_id"].endswith("-101")
    ]
    if not rows:
        return
    labels = [
        f"{r['predictor_label']}\n→ {MEASURES[r['outcome_symbol']].label.split(' (')[0]}"
        if r.get("outcome_symbol") in MEASURES
        else r["predictor_label"]
        for r in rows
    ]
    positions = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 0.62 * len(rows) + 2.0))
    for index, position in enumerate(positions):
        for key, offset, colour, name in (
            ("between", 0.16, "#1f4e79", "comparing children"),
            ("within", -0.16, "#c8791a", "within one child"),
        ):
            part = rows[index][key]
            ax.plot([part["lo"], part["hi"]], [position + offset] * 2, color=colour, linewidth=1.6)
            ax.plot([part["median"]], [position + offset], "o", color=colour, markersize=6)
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Association with the outcome level (log-odds per standard deviation)")
    ax.set_title("The same association, split two ways")
    handles = [
        plt.Line2D([], [], color="#1f4e79", marker="o", linestyle="-", label="comparing one child with another"),
        plt.Line2D([], [], color="#c8791a", marker="o", linestyle="-", label="within one child, wave to wave"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    save_styled_figure(str(out_dir), "pooled_levels_between_within", fig=fig, data=pd.DataFrame(rows))


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
        "design": build_design(art),
        "itt_suite": build_itt_suite(art),
        "triangulation": build_triangulation(art),
        "word_reading": build_word_reading(art),
        "mediation": build_mediation(art),
        "mechanism": build_mechanism(art),
        "pooled_levels": build_pooled_levels(art),
        "hearing_speech": build_hearing_speech(art),
        "gb": build_gb(art),
        "horseshoe": build_horseshoe(art),
        "ranking_agreement": build_ranking_agreement(art),
        "prior_dependence": build_prior_dependence(art),
        "blending_link": build_blending_link(art),
        "release": build_release_inventory(art),
    }

    figure_effects_forest(results, out_dir)
    figure_word_reading_envelope(results, out_dir)
    figure_mediation(results, out_dir)
    figure_pooled_levels(results, out_dir)
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
