# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Horseshoe global-shrinkage prior sensitivity for the ranking cross-checks (#382).

The prior-critical review (recommendation 6) asks the ``hs``/``rlm-hs`` family to
show its predictor ranking is stable over the regularised-horseshoe's tuning
choices — ``tau0`` (the expected-sparsity global scale) and ``slab_scale`` — since
that stability is what substantiates the "independent cross-check" framing; without
it the sub-top ranks rest on a verbal caveat only.

For each registered ranking model this script refits the same model in-process over
a one-at-a-time grid around the registered values (``tau0`` in {0.05, 0.2} at the
registered slab, ``slab_scale`` in {1.0, 4.0} at the registered ``tau0``), computes
the same ``P(|beta_k| > delta)`` ranking the pipeline reports, and compares each
cell with the model's **existing reporting fit** (the reference is not refit — the
cells are bound to it by its ``config.json``/``trace.nc`` hashes, following the ITT
sweep convention). Stability is summarised per cell as the Kendall rank
correlation, the top-3 overlap, and the maximum per-predictor change in
``P(|beta| > delta)``.

Outputs: one long per-predictor frame plus the per-cell stability summary in
``<output-root>/statistical_models/horseshoe_prior_sensitivity/``, and a
report-local copy of the summary beside each reference fit
(``horseshoe_prior_sensitivity.csv``). Every cell must pass the family convergence
gate (R-hat <= 1.01, ESS >= 400, BFMI >= 0.3, zero divergences) to count as
usable evidence; failing cells are recorded with ``converged=False`` and excluded
from the stability verdict.

This is deliberately a standalone, post-fit sensitivity sweep rather than an
in-fit ``run_subfit``. It starts from persisted primary artefacts, spans multiple
registered models and writes its own primary-fit hashes, sampling settings and
convergence record; ``run_subfit`` instead belongs to one live fit context and its
artifact manifest.

Usage::

    python scripts/horseshoe_prior_sensitivity.py                    # dev (fast)
    python scripts/horseshoe_prior_sensitivity.py --config reporting
    python scripts/horseshoe_prior_sensitivity.py --models lrp-rli-hs-001
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

import dse_research_utils.statistics.models.sampling as _sampling
from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import factories as _factories
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    spec_target_accept,
)
from language_reading_predictors.statistical_models.horseshoe import (
    HorseshoeRunPlan,
    resolve_horseshoe_run_plan,
)

DEFAULT_MODELS = (
    "lrp-rli-hs-001",
    "lrp-rli-hs-002",
    "lrp-rli-hs-003",
    "lrp-rli-hs-004",
    "lrp-rlm-hs-001",
)
TAU0_GRID = (0.05, 0.2)
SLAB_GRID = (1.0, 4.0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _spec_for(model_id: str) -> ModelSpec:
    module = model_id.replace("lrp-", "lrp_").replace("-", "_")
    mod = __import__(
        f"language_reading_predictors.statistical_models.{module}",
        fromlist=["SPEC"],
    )
    return mod.SPEC


def _build(plan: HorseshoeRunPlan, tau0: float, slab_scale: float):
    """Rebuild the registered model with the grid knobs; mirror the fit paths."""
    if plan.port == "rlm":
        from language_reading_predictors.statistical_models.preprocessing import (
            load_rlm_span_frame,
        )

        frame = load_rlm_span_frame(**plan.rlm_prepare_kwargs())
        kwargs = plan.rlm_factory_kwargs(predictors=list(frame.predictors))
        kwargs.update({"tau0": tau0, "slab_scale": slab_scale})
        return _factories.build_rlm_horseshoe_model(frame, **kwargs)
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    prepared = load_and_prepare(**plan.rli_prepare_kwargs())
    kwargs = plan.rli_factory_kwargs()
    kwargs.update({"tau0": tau0, "slab_scale": slab_scale})
    return _factories.build_horseshoe_model(prepared, **kwargs)


def _cell_converged(idata) -> tuple[bool, dict[str, float]]:
    # ``az.bfmi`` returns a DataTree in this environment; use the shared
    # per-chain helper the convergence gate itself uses.
    from language_reading_predictors.statistical_models.diagnostics import (
        _bfmi_per_chain,
    )

    div = int(idata.sample_stats["diverging"].values.sum())
    summ = az.summary(idata, kind="diagnostics")
    rhat = pd.to_numeric(summ["r_hat"], errors="coerce")
    ess_b = pd.to_numeric(summ["ess_bulk"], errors="coerce")
    ess_t = pd.to_numeric(summ["ess_tail"], errors="coerce")
    per_chain = _bfmi_per_chain(idata)
    bfmi = float(np.min(per_chain)) if per_chain else float("nan")
    stats = {
        "divergences": div,
        "max_r_hat": float(rhat.max()),
        "min_ess_bulk": float(ess_b.min()),
        "min_ess_tail": float(ess_t.min()),
        "min_bfmi": bfmi,
    }
    ok = (
        div == 0
        and stats["max_r_hat"] <= 1.01
        and stats["min_ess_bulk"] >= 400
        and stats["min_ess_tail"] >= 400
        and bfmi >= 0.3
    )
    return ok, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="dev")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument(
        "--target-accept",
        type=float,
        default=None,
        help=(
            "Override target_accept for every cell. The horseshoe funnel edge "
            "throws occasional single boundary divergences at the family's 0.99 "
            "(the zero-divergence gate then rejects an otherwise-perfect cell); "
            "0.999 clears them, as the registered hs-001 fit already documents."
        ),
    )
    args = ap.parse_args()

    sampling = _sampling.get_sampling_configuration(args.config, random_seed=20260701)
    out_root = Path(_paths.stat_dir())
    out_dir = out_root / "horseshoe_prior_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}  config={args.config}")

    long_rows: list[dict] = []
    cell_rows: list[dict] = []
    for model_id in args.models:
        spec = _spec_for(model_id)
        plan = resolve_horseshoe_run_plan(spec)
        delta = plan.delta
        ref_tau0 = plan.tau0
        ref_slab = plan.slab_scale
        # Reference = the model's existing reporting fit, bound by hashes.
        ref_dir = out_root / "models" / f"{model_id}-reporting"
        ref_rank_path = ref_dir / "predictor_ranking.csv"
        if not ref_rank_path.is_file():
            print(f"  {model_id}: SKIP — no reference reporting predictor_ranking.csv")
            continue
        ref = pd.read_csv(ref_rank_path)
        ref_order = list(ref["predictor"])
        ref_p = dict(zip(ref["predictor"], ref["p_abs_gt_delta"], strict=True))
        binding = {
            "primary_config_sha256": _sha256(ref_dir / "config.json"),
            "primary_trace_sha256": _sha256(ref_dir / "trace.nc"),
        }
        # target_accept: CLI override > registered spec override (hs-001 needs
        # 0.999) > family default.
        target_accept = (
            float(args.target_accept)
            if args.target_accept is not None
            else spec_target_accept(spec) or 0.99
        )

        cells = [(t, ref_slab) for t in TAU0_GRID] + [(ref_tau0, s) for s in SLAB_GRID]
        for tau0, slab in cells:
            print(f"  fitting {model_id}  tau0={tau0} slab_scale={slab} ...")
            built = _build(plan, tau0, slab)
            with built.model:
                idata = pm.sample(
                    draws=sampling.draws,
                    tune=sampling.tune,
                    chains=sampling.chains,
                    target_accept=target_accept,
                    nuts_sampler="nutpie",
                    random_seed=20260701,
                    progressbar=False,
                )
            ok, stats = _cell_converged(idata)
            ranking = _report.horseshoe_ranking(idata, delta=delta)
            order = list(ranking["predictor"])
            p_map = dict(zip(ranking["predictor"], ranking["p_abs_gt_delta"], strict=True))
            shared = [p for p in ref_order if p in p_map]
            ref_ranks = {p: i for i, p in enumerate(ref_order)}
            cell_ranks = {p: i for i, p in enumerate(order)}
            a = [ref_ranks[p] for p in shared]
            b = [cell_ranks[p] for p in shared]
            kendall = float(pd.Series(a).corr(pd.Series(b), method="kendall"))
            top3_ref, top3_cell = set(ref_order[:3]), set(order[:3])
            jacc = len(top3_ref & top3_cell) / len(top3_ref | top3_cell)
            max_dp = max(abs(p_map[p] - ref_p[p]) for p in shared)
            cell_rows.append(
                {
                    "model_id": model_id,
                    "outcome": spec.outcome_symbol,
                    "delta": delta,
                    "tau0": tau0,
                    "slab_scale": slab,
                    "reference_tau0": ref_tau0,
                    "reference_slab_scale": ref_slab,
                    "converged": ok,
                    **stats,
                    "kendall_tau_vs_reference": kendall,
                    "top3_jaccard_vs_reference": jacc,
                    "max_abs_delta_p_vs_reference": max_dp,
                    "sampling_draws": sampling.draws,
                    "sampling_tune": sampling.tune,
                    "sampling_chains": sampling.chains,
                    "sampling_target_accept": target_accept,
                    "config": args.config,
                    **binding,
                }
            )
            for _, r in ranking.iterrows():
                long_rows.append(
                    {
                        "model_id": model_id,
                        "tau0": tau0,
                        "slab_scale": slab,
                        "rank": int(r["rank"]),
                        "predictor": r["predictor"],
                        "p_abs_gt_delta": float(r["p_abs_gt_delta"]),
                        "beta_median": float(r["beta_median"]),
                        "converged": ok,
                    }
                )
            del idata

    cells_df = pd.DataFrame(cell_rows)
    long_df = pd.DataFrame(long_rows)
    cells_df.to_csv(out_dir / "horseshoe_prior_sensitivity.csv", index=False)
    long_df.to_csv(out_dir / "horseshoe_prior_sensitivity_rankings.csv", index=False)
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(
            {
                "config": args.config,
                "models": list(args.models),
                "tau0_grid": list(TAU0_GRID),
                "slab_grid": list(SLAB_GRID),
                "n_cells": len(cells_df),
            },
            fh,
            indent=2,
        )
    # Report-local copies beside each reference fit (reporting runs only —
    # a dev sweep must not sit beside a reporting fit looking authoritative).
    if args.config == "reporting":
        for model_id in cells_df["model_id"].unique():
            dst = out_root / "models" / f"{model_id}-reporting"
            if dst.is_dir():
                cells_df[cells_df["model_id"] == model_id].to_csv(
                    dst / "horseshoe_prior_sensitivity.csv", index=False
                )
                print(f"Wrote report-local sensitivity: {dst}")
    print(cells_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
