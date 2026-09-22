# SPDX-License-Identifier: AGPL-3.0-or-later
"""Objective-sensitivity check for four gradient-boosting models.

For each model and objective: (1) re-tune with scripts/tune_model.py under the
matched scoring metric (150 trials, seed 47, the #169 protocol), (2) fit the
tuned parameters at the reporting tier into a scratch output root. The
'registry' arm skips tuning and fits the committed MAE-tuned parameters as a
reproduction baseline. Resumable: completed stages are skipped.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["LRP_REPO"]).resolve()
N_TRIALS = int(os.environ.get("LRP_N_TRIALS", "150"))
SEED = 47

# Huber threshold delta = 1.345 * 1.4826 * MAD(y); nonword's MAD is zero (57 %
# zeros), so it falls back to 1.345 * mean |y - median| (target stats 2026-09-22).
HUBER_DELTA = {
    "lrp-rli-gbl-012": 12.96,
    "lrp-rli-gbg-012": 3.99,
    "lrp-rli-gbl-006": 17.95,
    "lrp-rli-gbl-013": 1.67,
}

MODELS_IN_ORDER = [
    "lrp-rli-gbl-012",  # word-reading level: skewed, floored, 11 trees under MAE
    "lrp-rli-gbl-013",  # nonword level: near-floor (57 % zeros)
    "lrp-rli-gbl-006",  # expressive vocabulary level: well-behaved
    "lrp-rli-gbg-012",  # word-reading gain: near-noise, can be negative
]

# objective name -> (lgbm objective, scoring, level-only?)
OBJECTIVES = {
    "mae": ("mae", "mae", False),
    "huber": ("huber", "rmse", False),
    "l2": ("regression", "rmse", False),
    "poisson": ("poisson", "rmse", True),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune(model_id: str, name: str) -> Path:
    objective, scoring, _ = OBJECTIVES[name]
    root = HERE / f"tune_{name}"
    best = root / "tuning" / model_id / "best_params.json"
    if best.exists():
        data = json.loads(best.read_text())
        if data.get("n_trials") == N_TRIALS and data["params"]["objective"] == objective:
            log(f"tune  {model_id} {name}: already done")
            return best
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "tune_model.py"),
        model_id,
        "--n-trials",
        str(N_TRIALS),
        "--scoring",
        scoring,
        "--lgbm-objective",
        objective,
        "--seed",
        str(SEED),
        "--output-dir",
        str(root),
    ]
    if name == "huber":
        cmd += ["--alpha", str(HUBER_DELTA[model_id])]
    log(f"tune  {model_id} {name}: start")
    t0 = time.time()
    with open(HERE / "logs" / f"tune_{model_id}_{name}.log", "w", encoding="utf-8") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
    log(f"tune  {model_id} {name}: done in {time.time() - t0:.0f}s")
    return best


def fit(model_id: str, name: str, params: dict | None) -> None:
    from language_reading_predictors import paths as _paths
    from language_reading_predictors.models.common import RunConfig
    from language_reading_predictors.models.registry import MODELS

    root = HERE / "fits" / name
    metrics = root / "models" / model_id / "metrics.json"
    if metrics.exists() and json.loads(metrics.read_text()).get("fit_complete"):
        log(f"fit   {model_id} {name}: already done")
        return
    cfg = MODELS[model_id]
    if params is not None:
        cfg = dataclasses.replace(cfg, model_params=dict(params))
    _paths.set_output_root(root)
    log(f"fit   {model_id} {name}: start (objective={cfg.model_params.get('objective')}, n_estimators={cfg.model_params.get('n_estimators')})")
    t0 = time.time()
    log_path = HERE / "logs" / f"fit_{model_id}_{name}.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = fh
        try:
            cfg.pipeline_cls(cfg, RunConfig.from_name("reporting")).fit()
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    log(f"fit   {model_id} {name}: done in {time.time() - t0:.0f}s")


def main() -> None:
    (HERE / "logs").mkdir(exist_ok=True)
    manifest = {"n_trials": N_TRIALS, "seed": SEED, "huber_delta": HUBER_DELTA, "runs": []}
    for model_id in MODELS_IN_ORDER:
        is_level = "-gbl-" in model_id
        fit(model_id, "registry", None)
        manifest["runs"].append({"model": model_id, "arm": "registry"})
        for name, (_, _, level_only) in OBJECTIVES.items():
            if level_only and not is_level:
                continue
            best = tune(model_id, name)
            params = json.loads(best.read_text())["params"]
            fit(model_id, name, params)
            manifest["runs"].append({"model": model_id, "arm": name, "best_params": str(best)})
        (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("ALL DONE")


if __name__ == "__main__":
    main()
