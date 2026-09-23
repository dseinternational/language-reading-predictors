# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare the 22 September MAE fits (backup) with the Huber refit, all 50 models.

Writes output/tuning/huber_vs_mae_fits.csv and a Markdown summary alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(r"V:\dev\dseinternational\language-reading-predictors")
OLD = ROOT / "output" / "models.pre-huber-20260922"
NEW = ROOT / "output" / "models"
OUT_CSV = ROOT / "output" / "tuning" / "huber_vs_mae_fits.csv"
OUT_MD = ROOT / "output" / "tuning" / "huber_vs_mae_fits.md"
TOP_K = 5


def load(d: Path) -> dict | None:
    m = d / "metrics.json"
    if not m.exists():
        return None
    metrics = json.loads(m.read_text())
    if not metrics.get("fit_complete"):
        return None
    cfg = json.loads((d / "config.json").read_text())
    perm = pd.read_csv(d / "permutation_importance.csv")
    perm["z"] = perm["importance_mean"] / perm["importance_std"].replace(0, np.nan)
    stab = pd.read_csv(d / "stability_selection.csv")
    shap = pd.read_csv(d / "shap_direction_diagnostics.csv")
    return {
        "metrics": metrics,
        "params": cfg.get("model_params", {}),
        "commit": (cfg.get("provenance") or {}).get("source", {}).get("commit"),
        "dirty": (cfg.get("provenance") or {}).get("source", {}).get("dirty"),
        "perm": perm,
        "stab": stab,
        "shap": shap,
    }


def replicated(run: dict) -> set[str]:
    z2 = set(run["perm"].loc[run["perm"]["z"] >= 2, "feature"])
    app = run["stab"].set_index("feature")["appearance_rate_top_k"]
    return {f for f in z2 if app.get(f, 0.0) >= 0.5}


def sign_of(run: dict, f: str) -> str:
    sh = run["shap"].set_index("feature")
    sp = sh["feature_shap_spearman"].get(f, np.nan)
    if np.isnan(sp):
        return "?"
    return "+" if sp > 0 else "-"


rows = []
for d in sorted(NEW.glob("lrp-rli-gb*")):
    model = d.name
    new = load(d)
    old = load(OLD / model)
    if new is None:
        rows.append({"model_id": model, "status": "new fit missing or incomplete"})
        continue
    row = {
        "model_id": model,
        "status": "ok",
        "new_commit": (new["commit"] or "")[:8],
        "new_dirty": new["dirty"],
        "new_objective": new["params"].get("objective"),
        "alpha": new["params"].get("alpha"),
        "new_trees": new["params"].get("n_estimators"),
        "new_oof_r2": new["metrics"].get("cv_pooled_r2"),
        "new_oof_rmse": new["metrics"].get("cv_pooled_rmse"),
        "new_oof_mae": new["metrics"].get("cv_pooled_mae"),
        "new_in_sample_r2": new["metrics"].get("in_sample_r2"),
        "new_top5": ", ".join(new["perm"].head(TOP_K)["feature"]),
        "new_top1_sign": sign_of(new, new["perm"].iloc[0]["feature"]),
        "new_n_z2": int((new["perm"]["z"] >= 2).sum()),
        "new_replicated": ", ".join(sorted(replicated(new))) or "-",
    }
    if old is not None:
        rho = spearmanr(
            new["perm"].set_index("feature")["importance_mean"].sort_index(),
            old["perm"].set_index("feature")["importance_mean"].sort_index(),
        ).correlation
        old_top = list(old["perm"].head(TOP_K)["feature"])
        new_top = list(new["perm"].head(TOP_K)["feature"])
        flips = []
        for f in set(new["perm"].loc[new["perm"]["z"] >= 2, "feature"]) | set(
            old["perm"].loc[old["perm"]["z"] >= 2, "feature"]
        ):
            if sign_of(new, f) != sign_of(old, f):
                flips.append(f"{f}:{sign_of(old, f)}>{sign_of(new, f)}")
        row.update(
            {
                "old_trees": old["params"].get("n_estimators"),
                "old_oof_r2": old["metrics"].get("cv_pooled_r2"),
                "old_oof_rmse": old["metrics"].get("cv_pooled_rmse"),
                "old_oof_mae": old["metrics"].get("cv_pooled_mae"),
                "old_in_sample_r2": old["metrics"].get("in_sample_r2"),
                "d_oof_r2": row["new_oof_r2"] - old["metrics"].get("cv_pooled_r2"),
                "d_oof_rmse": row["new_oof_rmse"] - old["metrics"].get("cv_pooled_rmse"),
                "d_oof_mae": row["new_oof_mae"] - old["metrics"].get("cv_pooled_mae"),
                "rho_perm": rho,
                "top1_same": old_top[0] == new_top[0],
                "top5_overlap": len(set(old_top) & set(new_top)),
                "old_top5": ", ".join(old_top),
                "old_replicated": ", ".join(sorted(replicated(old))) or "-",
                "sign_flips_z2": "; ".join(sorted(flips)) or "-",
            }
        )
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
ok = df[df["status"] == "ok"]
lines = ["# Huber refit against the 22 September MAE fits\n"]
if "d_oof_r2" in ok:
    for fam, sub in ok.groupby(ok["model_id"].str.contains("-gbg-").map({True: "gain", False: "level"})):
        lines.append(f"## {fam} models (n={len(sub)})\n")
        lines.append(
            f"- OOF R2 change: median {sub['d_oof_r2'].median():+.3f}, "
            f"min {sub['d_oof_r2'].min():+.3f}, max {sub['d_oof_r2'].max():+.3f}; "
            f"improved in {(sub['d_oof_r2'] > 0).sum()}/{len(sub)}"
        )
        lines.append(
            f"- OOF RMSE change: median {sub['d_oof_rmse'].median():+.3f}; lower in {(sub['d_oof_rmse'] < 0).sum()}/{len(sub)}"
        )
        lines.append(
            f"- OOF MAE change: median {sub['d_oof_mae'].median():+.3f}; lower in {(sub['d_oof_mae'] < 0).sum()}/{len(sub)}"
        )
        lines.append(
            f"- Spearman rho of permutation importance: median {sub['rho_perm'].median():.2f}, "
            f"min {sub['rho_perm'].min():.2f}; top predictor unchanged in {sub['top1_same'].sum()}/{len(sub)}; "
            f"median top-5 overlap {sub['top5_overlap'].median():.0f}"
        )
        lines.append(
            f"- Trees: old median {sub['old_trees'].median():.0f}, new median {sub['new_trees'].median():.0f}\n"
        )
    lines.append("## Per model\n")
    cols = [
        "model_id", "old_trees", "new_trees", "old_oof_r2", "new_oof_r2", "old_oof_rmse", "new_oof_rmse",
        "rho_perm", "top1_same", "top5_overlap", "old_top5", "new_top5", "sign_flips_z2",
    ]
    show = ok[cols].copy()
    for c in ["old_oof_r2", "new_oof_r2", "old_oof_rmse", "new_oof_rmse"]:
        show[c] = show[c].round(3)
    show["rho_perm"] = show["rho_perm"].round(2)
    lines.append(show.to_markdown(index=False))
text = "\n".join(lines)
OUT_MD.write_text(text, encoding="utf-8")
print(text)
print(f"\nwritten {OUT_CSV} and {OUT_MD}; incomplete: {(df['status'] != 'ok').sum()}")
