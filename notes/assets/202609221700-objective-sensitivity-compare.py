# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare the objective-sensitivity fits: fit quality, ranking agreement, direction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ARMS = ["registry", "mae", "huber", "l2", "poisson"]
MODELS = ["lrp-rli-gbl-012", "lrp-rli-gbl-013", "lrp-rli-gbl-006", "lrp-rli-gbg-012"]
REF = "mae"
TOP_K = 5


def load(model: str, arm: str) -> dict | None:
    d = HERE / "fits" / arm / "models" / model
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
    params = cfg.get("model_params") or cfg.get("params") or {}
    return {"metrics": metrics, "cfg": cfg, "params": params, "perm": perm, "stab": stab, "shap": shap}


def replicated(run: dict) -> set[str]:
    z2 = set(run["perm"].loc[run["perm"]["z"] >= 2, "feature"])
    app = run["stab"].set_index("feature")["appearance_rate_top_k"]
    return {f for f in z2 if app.get(f, 0.0) >= 0.5}


def main() -> None:
    out = []
    for model in MODELS:
        runs = {arm: load(model, arm) for arm in ARMS}
        runs = {k: v for k, v in runs.items() if v is not None}
        if not runs:
            continue
        out.append(f"\n## {model}\n")
        # --- fit quality
        rows = []
        for arm, r in runs.items():
            m, p = r["metrics"], r["params"]
            rows.append(
                {
                    "arm": arm,
                    "objective": p.get("objective"),
                    "trees": p.get("n_estimators"),
                    "leaves": p.get("num_leaves"),
                    "lr": round(float(p.get("learning_rate", np.nan)), 3),
                    "OOF MAE": round(m["cv_pooled_mae"], 2),
                    "OOF RMSE": round(m["cv_pooled_rmse"], 2),
                    "OOF R2": round(m["cv_pooled_r2"], 3),
                    "OOF MedAE": round(m["cv_pooled_medae"], 2),
                    "in-sample R2": round(m["in_sample_r2"], 3),
                }
            )
        out.append("### Fit quality (pooled out-of-fold)\n")
        out.append(pd.DataFrame(rows).to_markdown(index=False))
        # --- ranking agreement vs REF
        ref = runs.get(REF)
        rows = []
        for arm, r in runs.items():
            perm = r["perm"].set_index("feature")["importance_mean"]
            top = list(r["perm"].head(TOP_K)["feature"])
            rep = replicated(r)
            row = {
                "arm": arm,
                "top-5": ", ".join(top),
                "n z>=2": int((r["perm"]["z"] >= 2).sum()),
                "replicated": ", ".join(sorted(rep)) or "-",
            }
            if ref is not None:
                rp = ref["perm"].set_index("feature")["importance_mean"]
                common = perm.index.intersection(rp.index)
                rho = spearmanr(perm.loc[common], rp.loc[common]).correlation
                row["rho vs mae"] = round(float(rho), 2)
                row["top-5 overlap"] = len(set(top) & set(ref["perm"].head(TOP_K)["feature"]))
                row["replicated overlap"] = f"{len(rep & replicated(ref))}/{len(rep | replicated(ref))}"
            rows.append(row)
        out.append("\n### Ranking agreement (permutation importance, vs the re-tuned MAE arm)\n")
        out.append(pd.DataFrame(rows).to_markdown(index=False))
        # --- direction of any predictor in a top-5 anywhere
        union = []
        for r in runs.values():
            for f in r["perm"].head(TOP_K)["feature"]:
                if f not in union:
                    union.append(f)
        rows = []
        for f in union:
            row = {"feature": f}
            for arm, r in runs.items():
                rank = r["perm"].reset_index(drop=True)
                pos = rank.index[rank["feature"] == f]
                sh = r["shap"].set_index("feature")
                sp = sh["feature_shap_spearman"].get(f, np.nan)
                flag = sh["shape_flag"].get(f, "")
                z = float(rank.loc[pos[0], "z"]) if len(pos) else np.nan
                row[arm] = f"#{pos[0] + 1 if len(pos) else '-'} z={z:.1f} {'+' if sp > 0 else '-' if sp < 0 else '?'}{'' if 'monotonic' in str(flag) and 'non' not in str(flag) else '~'}"
            rows.append(row)
        out.append("\n### Rank, permutation z and SHAP direction for every predictor that is top-5 in any arm (~ = noisy or non-monotonic)\n")
        out.append(pd.DataFrame(rows).to_markdown(index=False))
    text = "\n".join(out)
    (HERE / "comparison.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
