# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rank predictors on the full set (issue #116, Phase 1).

Produces an ordered, **cluster-first**, stability-annotated candidate list for an
outcome by fitting one LightGBM on the *full* ``DEFAULT_GAIN`` / ``DEFAULT_LEVEL``
set (no pruning), reusing the existing ``EstimatorPipeline`` stages. This is the
ranking counterpart to the selection pipeline: it does not prune to a subset, it
ranks and groups every candidate and reports the uncertainty.

Design decisions baked in:

* **Cluster level is the primary unit.** Correlated predictors are clustered on the
  distance-correlation matrix and ranked *as groups* by joint (grouped) out-of-fold
  permutation importance — see ``cluster_ranking.csv``. Per-feature scores
  (``predictor_ranking.csv``) are within-cluster detail. The reason: per-feature
  permutation z is highly sensitive to ``cv_splits`` (in the pilot, ``lrpgbl06`` `b1exto`
  z ran 1.9 → 1.5 → 0.33 at cv = 5 / 10 / 51), whereas the cluster-level ordering is
  stable. Read clusters first.
* **The cross-validation config is pinned and recorded.** ``--cv-splits`` and
  ``--perm-repeats`` default to the model's own registered values (the reporting
  protocol), so the ranking is consistent with how the project evaluates everything
  else rather than using an ad-hoc number. The config used is written to
  ``ranking_meta.json``. ``--quick`` drops to a fast dev tier.
* **Same-skill contamination is a curated annotation, not a prune.** ``SAME_SKILL_SIBLINGS``
  flags predictors that are concurrent restatements of the *outcome* (a predictor↔outcome
  problem clustering cannot catch). The flag rides along in every artefact, and an
  "excluding same-skill" ranking view is emitted for outcomes that have a sibling.

Reuses ``EstimatorPipeline`` stages directly (not ``fit()``, which gates
clustering/SHAP/stability behind run tiers and ends with a Quarto ``report()`` that
no ad-hoc model id has). ``GroupKFold`` by ``subject_id``, seed 47.

Usage::

    python scripts/rank_predictors.py --model lrpgbl06          # reporting-fidelity
    python scripts/rank_predictors.py --model lrpgbg12 --quick   # fast dev tier
    python scripts/rank_predictors.py --model lrpgbg05 --cv-splits 10 --cutoff 0.4

Artefacts land in ``output/ranking/<model>/`` (gitignored): ``cluster_ranking.csv``
(primary), ``predictor_ranking.csv``, ``cluster_cutoff_sensitivity.csv``,
``ranking_vs_selected.csv``, ``ranking_excluding_same_skill.csv`` (outcomes with a
sibling), ``ranking_meta.json``, ``distance_corr_dendrogram.png`` and
``shap_summary.png`` (beeswarm). The downstream consumer of this ranking is
``scripts/predictability_readout.py --from-ranking``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import root_mean_squared_error

from language_reading_predictors.data_variables import Predictors
from language_reading_predictors.models.base_pipeline import _OUTPUT_DIR, _clear_directory
from language_reading_predictors.models.cluster_ranking import (
    SAME_SKILL_SIBLINGS,
    assemble_ranking,
    cluster_ranking_table,
)
from language_reading_predictors.models.common import RunConfig
from language_reading_predictors.models.registry import MODELS
from language_reading_predictors.stats_utils import distance_corr_matrix


_ROOT = Path(__file__).resolve().parent.parent
RANKING_ROOT = _ROOT / "output" / "ranking"  # gitignored
# Helper fits go wherever the *installed* pipeline writes — base_pipeline._OUTPUT_DIR,
# which resolves to the editable install's repo root, NOT necessarily this script's
# worktree. Use that exact dir so scratch cleanup targets the real fits.
MODELS_ROOT = _OUTPUT_DIR


def _fmt(x, spec: str = "{:.3f}") -> str:
    """Format a metric that may be ``None``/``NaN``.

    ``pooled_r2`` is ``None`` when ``ss_tot == 0`` (``base_pipeline.cross_validate``)
    and ``max_z`` is ``NaN`` when every per-feature SD is 0 — either would otherwise
    crash an f-string ``{:.3f}`` or silently misformat.
    """
    return "n/a" if x is None or pd.isna(x) else spec.format(x)


# ── config construction ────────────────────────────────────────────────────────
def make_config(model_id: str, *, kind: str):
    """Return an ad-hoc ``ModelConfig``.

    kind="full"    -> full DEFAULT_* set minus target (no pruning)
    kind="noskill" -> full set minus target minus curated same-skill siblings
    kind="sel"     -> the registered (currently selected) predictor set
    """
    base = MODELS[model_id]
    target = base.target_var
    full = Predictors.DEFAULT_GAIN if target.endswith("_gain") else Predictors.DEFAULT_LEVEL
    siblings = SAME_SKILL_SIBLINGS.get(target, [])

    if kind == "sel":
        preds = list(base.predictor_vars)
    else:
        preds = [p for p in full if p != target]  # leakage guard
        if kind == "noskill":
            preds = [p for p in preds if p not in siblings]

    cfg = dataclasses.replace(
        base,
        model_id=f"rank_{model_id}_{kind}",
        predictor_vars=preds,
        selection_history=[],
        pdp_features=None,
        shap_scatter_specs=[],
    )
    return cfg, target, siblings


def make_run(base_cfg, *, cv_splits, perm_repeats, quick):
    """Pinned, recorded run config. Defaults inherit the model's reporting protocol."""
    if quick:
        return RunConfig(name="rank-quick", n_estimators=None, cv_splits=5,
                         perm_importance_repeats=5, skip_shap=False, skip_pdp=True,
                         skip_correlation=False)
    return RunConfig(
        name="rank",
        n_estimators=None,
        cv_splits=cv_splits if cv_splits is not None else base_cfg.cv_splits,
        perm_importance_repeats=perm_repeats if perm_repeats is not None
        else base_cfg.perm_importance_repeats,
        skip_shap=False, skip_pdp=True, skip_correlation=False,
    )


def run_stages(cfg, run, *, cluster_cutoff=0.4, do_cluster=True, do_shap=True, do_stability=True):
    """Instantiate the pipeline and drive the stages we need, in dependency order."""
    pipe = cfg.pipeline_cls(cfg, run)
    ctx = pipe.context
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_directory(ctx.output_dir)  # parity with fit(): clears files AND subdirs

    pipe.prepare_data()
    pipe.configure_model()
    pipe.cross_validate()              # -> ctx.cv_results, ctx.pooled_cv_metrics
    pipe.fit_model()                   # full-data fit (needed for SHAP)
    pipe.evaluate()                    # -> ctx.eval_df (shap_analysis reads it)
    pipe.permutation_importance_analysis()  # -> ctx.perm_importance_df
    if do_cluster:
        pipe.feature_selection_diagnostics(cluster_cutoff=cluster_cutoff)  # dendrogram, cluster_table
    if do_shap:
        pipe.shap_analysis()           # -> ctx.shap_values, shap_summary.png (beeswarm)
        pipe.shap_direction_diagnostics()  # -> shap_direction_diagnostics.csv
    if do_stability:
        pipe.stability_selection()     # -> stability_selection.csv (appearance_rate_top_k)
    return pipe


# ── cluster-level (grouped) permutation importance ─────────────────────────────
def _grouped_perm_deltas(estimators, X, y, test_indices, cluster_cols, *, n_repeats, seed):
    """Joint (grouped) out-of-fold permutation deltas, one block per cluster.

    Pure-numeric core of :func:`cluster_permutation_importance`, split out so the
    headline grouped-shuffle metric is unit-testable without a pipeline. For each
    held-out fold it permutes ALL of a cluster's columns together (one row-permutation
    per repeat applied to the whole block), which removes the within-cluster
    substitution dilution that deflates per-feature scores. The RNG is reset to
    ``seed`` at the start of each fold, mirroring ``permutation_importance_analysis``
    (which passes ``random_state=cfg.random_seed`` per fold). ``cluster_cols`` maps
    cluster id -> column *positions* in ``X``. Returns cluster id -> array of deltas
    (held-out RMSE rise when the block is permuted; positive = the cluster was useful).
    """
    y = np.asarray(y, dtype=float)
    deltas: dict[int, list[float]] = {c: [] for c in cluster_cols}
    for est, val_idx in zip(estimators, test_indices):
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]
        base_rmse = root_mean_squared_error(y_val, est.predict(X_val))
        n = len(X_val)
        rng = np.random.default_rng(seed)  # reset per fold (matches the per-feature loop)
        for c, cols in cluster_cols.items():
            for _ in range(n_repeats):
                perm = rng.permutation(n)
                Xp = X_val.copy()
                Xp.iloc[:, cols] = X_val.iloc[perm, cols].to_numpy()  # joint block shuffle
                deltas[c].append(root_mean_squared_error(y_val, est.predict(Xp)) - base_rmse)
    return {c: np.asarray(v) for c, v in deltas.items()}


def cluster_permutation_importance(pipe, clusters_by_feature, *, n_repeats):
    """Joint/grouped out-of-fold permutation importance, one block per cluster.

    Thin wrapper over :func:`_grouped_perm_deltas` (the pure, unit-tested core) that
    pulls the per-fold estimators, held-out indices and design matrix off the pipeline.
    """
    ctx = pipe.context
    cfg = ctx.config
    cv = ctx.cv_results
    feats = list(ctx.X.columns)
    cluster_ids = sorted({clusters_by_feature[f] for f in feats})
    cluster_cols = {
        c: [i for i, f in enumerate(feats) if clusters_by_feature[f] == c]
        for c in cluster_ids
    }

    deltas = _grouped_perm_deltas(
        cv["estimator"], ctx.X, ctx.y, cv["indices"]["test"], cluster_cols,
        n_repeats=n_repeats, seed=cfg.random_seed,
    )

    rows = []
    for c in cluster_ids:
        arr = deltas[c]
        members = sorted(f for f in feats if clusters_by_feature[f] == c)
        rows.append({
            "cluster_id": c,
            "cluster_perm_imp_mean": float(arr.mean()),
            "cluster_perm_imp_sd": float(arr.std()),
            "n_members": len(members),
            "members": ",".join(members),
        })
    out = pd.DataFrame(rows).sort_values("cluster_perm_imp_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "cluster_rank", np.arange(1, len(out) + 1))
    return out


# ── cluster cut-height sensitivity ──────────────────────────────────────────────
def ward_linkage(X):
    """Replicate the linkage ``feature_selection_diagnostics`` builds (dcor on mean-filled X)."""
    Xf = X.replace({pd.NA: np.nan}).astype("float64")
    Xf = Xf.fillna(Xf.mean())
    dcm = distance_corr_matrix(Xf)
    dissim = 1.0 - dcm
    np.fill_diagonal(dissim, 0.0)
    np.clip(dissim, 0.0, 1.0, out=dissim)
    return hierarchy.ward(squareform(dissim, checks=False))


def cutoff_sensitivity(X, target, cutoffs=(0.2, 0.3, 0.4, 0.5, 0.6)):
    """How cluster membership moves with the dendrogram cut height (the #116 trade-off:
    the 0.70 *delete* threshold becomes a cut-height *grouping* choice)."""
    Z = ward_linkage(X)
    feats = list(X.columns)
    vocab_markers = ["b1exto", "eowpvt", "aptinfo", "rowpvt", "b1reto", "aptgram"]
    siblings = SAME_SKILL_SIBLINGS.get(target, [])
    anchor = next((m for m in (siblings + vocab_markers) if m in feats), feats[0])
    rows = []
    for t in cutoffs:
        cl = hierarchy.fcluster(Z, t=t, criterion="distance")
        memb = {f: int(c) for f, c in zip(feats, cl)}
        anchor_members = sorted(f for f in feats if memb[f] == memb[anchor])
        rows.append({
            "cutoff": t,
            "n_clusters": int(len(set(cl))),
            "anchor": anchor,
            "anchor_cluster_size": len(anchor_members),
            "anchor_cluster_members": ",".join(anchor_members),
        })
    return pd.DataFrame(rows)


# ── ranking assembly ────────────────────────────────────────────────────────────
def ranking_vs_selected(model_id, full_pipe, sel_pipe):
    """Side-by-side: each predictor's full-set rank vs whether selection kept it."""
    selected = list(MODELS[model_id].predictor_vars)
    full = full_pipe.context.perm_importance_df.reset_index(drop=True).copy()
    full["full_rank"] = np.arange(1, len(full) + 1)
    full["in_selected_set"] = full["feature"].isin(selected)
    sel = sel_pipe.context.perm_importance_df.reset_index(drop=True).copy()
    sel["selected_rank"] = np.arange(1, len(sel) + 1)
    sel = sel[["feature", "selected_rank", "importance_mean"]].rename(
        columns={"importance_mean": "selected_imp"}
    )
    out = full.merge(sel, on="feature", how="left").rename(columns={"importance_mean": "full_imp"})
    return out[["full_rank", "feature", "full_imp", "in_selected_set", "selected_rank", "selected_imp"]]


def conditional_dropout_check(model_id, run, cluster_imp, clusters_by_feature, base_full_pipe):
    """Conditional cross-check on the dominant cluster: drop the whole top cluster and
    report the pooled OOF R² change (how much the cluster *jointly* buys)."""
    feats = list(base_full_pipe.context.X.columns)
    top_cluster = int(cluster_imp.iloc[0]["cluster_id"])
    top_members = [f for f in feats if clusters_by_feature[f] == top_cluster]
    cfg, _, _ = make_config(model_id, kind="full")
    cfg = dataclasses.replace(
        cfg,
        model_id=f"rank_{model_id}_drop_top_cluster",
        predictor_vars=[p for p in cfg.predictor_vars if p not in top_members],
    )
    pipe = cfg.pipeline_cls(cfg, run)
    pipe.context.output_dir.mkdir(parents=True, exist_ok=True)
    pipe.prepare_data()
    pipe.configure_model()
    pipe.cross_validate()
    return {
        "top_cluster_members": top_members,
        "r2_full": base_full_pipe.context.pooled_cv_metrics.get("pooled_r2"),
        "r2_drop_top_cluster": pipe.context.pooled_cv_metrics.get("pooled_r2"),
    }


# ── orchestration ───────────────────────────────────────────────────────────────
def run_model(model_id, *, cutoff=0.4, cv_splits=None, perm_repeats=None, quick=False):
    if model_id not in MODELS:
        raise SystemExit(f"unknown model id {model_id!r}; known: {sorted(MODELS)}")
    base = MODELS[model_id]
    run = make_run(base, cv_splits=cv_splits, perm_repeats=perm_repeats, quick=quick)
    out = RANKING_ROOT / model_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\n  RANK PREDICTORS — {model_id}  "
          f"(cutoff={cutoff}, cv_splits={run.cv_splits}, quick={quick})\n{'=' * 78}")

    try:
        # 1. full-set fit (the ranking)
        cfg_full, target, siblings = make_config(model_id, kind="full")
        full = run_stages(cfg_full, run, cluster_cutoff=cutoff, do_stability=not quick)
        clusters_tbl = pd.read_csv(full.context.output_dir / "cluster_table.csv")
        clusters_by_feature = dict(zip(clusters_tbl["feature"], clusters_tbl["cluster_id"]))
        cluster_imp = cluster_permutation_importance(
            full, clusters_by_feature, n_repeats=5 if quick else run.perm_importance_repeats)

        ranking = assemble_ranking(full, target, siblings, cluster_imp)
        ranking.to_csv(out / "predictor_ranking.csv", index=False)
        cluster_rank = cluster_ranking_table(cluster_imp, ranking, siblings)
        cluster_rank.to_csv(out / "cluster_ranking.csv", index=False)  # PRIMARY artefact

        # 2. sensitivity to cut height
        sens = cutoff_sensitivity(full.context.X, target)
        sens.to_csv(out / "cluster_cutoff_sensitivity.csv", index=False)

        # 3. side-by-side vs the currently-selected set
        cfg_sel, _, _ = make_config(model_id, kind="sel")
        sel = run_stages(cfg_sel, run, do_cluster=False, do_shap=False, do_stability=False)
        rvs = ranking_vs_selected(model_id, full, sel)
        rvs.to_csv(out / "ranking_vs_selected.csv", index=False)

        # 4. conditional cross-check on the dominant cluster
        cond = conditional_dropout_check(model_id, run, cluster_imp, clusters_by_feature, full)

        # 5. same-skill handling: excluding-siblings view + curated-variant compare
        noskill = None
        if siblings:
            cfg_ns, _, _ = make_config(model_id, kind="noskill")
            ns = run_stages(cfg_ns, run, cluster_cutoff=cutoff, do_stability=False)
            ns.context.perm_importance_df.to_csv(out / "ranking_excluding_same_skill.csv", index=False)
            noskill = {"dropped": siblings,
                       "top5": ns.context.perm_importance_df.head(5)["feature"].tolist(),
                       "r2_noskill": ns.context.pooled_cv_metrics.get("pooled_r2")}
            nc_id = f"{model_id}_noconstruct"
            if nc_id in MODELS:
                nc_cfg = dataclasses.replace(MODELS[nc_id], model_id=f"rank_{nc_id}",
                                             selection_history=[], pdp_features=None,
                                             shap_scatter_specs=[])
                nc = run_stages(nc_cfg, run, do_cluster=False, do_shap=False, do_stability=False)
                noskill["curated_variant"] = nc_id
                noskill["r2_curated_variant"] = nc.context.pooled_cv_metrics.get("pooled_r2")

        # self-describing metadata (records the pinned cv config)
        meta = {
            "model_id": model_id, "target": target,
            "n_observations": int(len(full.context.X)),
            "n_children": int(full.context.groups.nunique()),
            "full_set_size": int(len(full.context.X.columns)),
            "cv_splits": run.cv_splits, "perm_repeats": run.perm_importance_repeats,
            "cluster_cutoff": cutoff, "random_seed": base.random_seed,
            "pooled_oof_r2": full.context.pooled_cv_metrics.get("pooled_r2"),
            "primary_artefact": "cluster_ranking.csv",
            "note": ("cluster-level grouped importance is the primary unit; per-feature z "
                     "is cv_splits-sensitive (read clusters first)"),
        }
        (out / "ranking_meta.json").write_text(json.dumps(meta, indent=2))

        for fn in ("distance_corr_dendrogram.png", "shap_summary.png"):
            src = full.context.output_dir / fn
            if src.exists():
                shutil.copy(src, out / fn)

        _print_summary(model_id, target, siblings, full, ranking, cluster_rank, sens,
                       cond, noskill, out)
        return ranking
    finally:
        # The helper fits land in output/models/rank_<id>_* (full / sel / noskill /
        # drop_top_cluster / noconstruct) because EstimatorPipeline derives output_dir
        # from model_id. They are scratch — never real models — but upload.py globs every
        # output/models/ subdir, so leaving them would publish them as if fitted. Remove
        # them here (success or failure); the ranking artefacts are under output/ranking/<id>/.
        for d in MODELS_ROOT.glob(f"rank_{model_id}_*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


def _print_summary(model_id, target, siblings, full, ranking, cluster_rank, sens,
                   cond, noskill, out):
    r2 = full.context.pooled_cv_metrics.get("pooled_r2")
    max_z = ranking["z"].abs().max()
    print(f"\n  outcome={target}  n_obs={len(full.context.X)}  "
          f"n_children={full.context.groups.nunique()}  "
          f"full_set={len(full.context.X.columns)}  pooled_OOF_R2={_fmt(r2)}")
    if pd.isna(max_z):
        verdict = "PLATEAU: all per-feature SD = 0 (z undefined)"
    else:
        verdict = "PLATEAU: none > 1 SD above 0" if max_z < 1 else "top feature clears z=1"
    print(f"  max |z| (per-feature perm imp / SD) = {_fmt(max_z, '{:.2f}')}  ({verdict})")
    print("\n  PRIMARY — cluster ranking (grouped permutation importance):")
    for _, r in cluster_rank.head(4).iterrows():
        skill = " *same-skill present*" if r["any_same_skill"] else ""
        print(f"    #{int(r['cluster_rank'])}  imp={r['cluster_perm_imp_mean']:+.4f}  "
              f"rep={r['representative']}  [{r['members']}]{skill}")
    if siblings:
        flagged = ranking.loc[ranking["same_skill_of_outcome"], "member"].tolist()
        print(f"\n  same-skill flag -> {flagged} (curated siblings of {target})")
        if noskill:
            line = f"    excluding same-skill top5: {noskill['top5']}  R2={_fmt(noskill['r2_noskill'])}"
            if "r2_curated_variant" in noskill:
                line += f"  | curated {noskill['curated_variant']} R2={_fmt(noskill['r2_curated_variant'])}"
            print(line)
    print(f"\n  conditional check: drop dominant cluster {cond['top_cluster_members']}")
    print(f"    pooled R2  full={_fmt(cond['r2_full'])} -> drop-top-cluster={_fmt(cond['r2_drop_top_cluster'])}")
    print("\n  cut-height sensitivity:")
    print(sens[["cutoff", "n_clusters", "anchor_cluster_size"]].to_string(index=False))
    print(f"\n  artefacts -> {out}/  (primary: cluster_ranking.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank predictors on the full set (issue #116, Phase 1).")
    ap.add_argument("--model", required=True, help="model id, e.g. lrpgbl06, lrpgbg12, lrpgbg05")
    ap.add_argument("--cutoff", type=float, default=0.4, help="dendrogram cut height (default 0.4)")
    ap.add_argument("--cv-splits", type=int, default=None,
                    help="GroupKFold splits (default: the model's registered cv_splits)")
    ap.add_argument("--perm-repeats", type=int, default=None,
                    help="permutation-importance repeats (default: the model's registered value)")
    ap.add_argument("--quick", action="store_true",
                    help="fast dev tier: cv=5, repeats=5, skip stability")
    args = ap.parse_args()
    run_model(args.model, cutoff=args.cutoff, cv_splits=args.cv_splits,
              perm_repeats=args.perm_repeats, quick=args.quick)


if __name__ == "__main__":
    main()
