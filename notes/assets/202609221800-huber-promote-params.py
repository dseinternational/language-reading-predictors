# SPDX-License-Identifier: AGPL-3.0-or-later
"""Promote Huber-tuned parameters from output/tuning/*/best_params.json into the
50 model modules. Preserves each module's key schema (random_state where
present), inserts ``alpha`` after ``objective``, renames ``_LGBM_MAE_PARAMS`` to
``_LGBM_HUBER_PARAMS`` and rewrites the MAE-tuned prose.

Usage: python promote_params.py [--dry-run]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(r"V:\dev\dseinternational\language-reading-predictors")
MODULES = ROOT / "src" / "language_reading_predictors" / "models"
TUNING = ROOT / "output" / "tuning"
DRY = "--dry-run" in sys.argv

PARAM_KEYS = [
    "objective",
    "alpha",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "n_jobs",
    "verbosity",
    "random_state",
]

PROSE = [
    ("MAE-tuned", "Huber-tuned"),
    ('"objective": "mae"', '"objective": "huber"'),
    ("_LGBM_MAE_PARAMS", "_LGBM_HUBER_PARAMS"),
    ("hyperparameters (MAE-tuned)", "hyperparameters (Huber-tuned)"),
]

BLOCK_RE = re.compile(
    r"(_LGBM_(?:MAE|HUBER)_PARAMS: dict\[str, float \| int \| str\] = \{\n)(.*?)(\n\})",
    re.DOTALL,
)
KEY_RE = re.compile(r'^\s*"([a-z_]+)": (.+?),\s*$')


def fmt(v):
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


changed = 0
for path in sorted(MODULES.glob("lrp_rli_gb*_*.py")):
    model_id = path.stem.replace("_", "-").replace("lrp-rli-", "lrp-rli-")
    model_id = "lrp-rli-" + path.stem[len("lrp_rli_"):].replace("_", "-")
    bp_path = TUNING / model_id / "best_params.json"
    if not bp_path.is_file():
        print(f"SKIP {model_id}: no best_params.json")
        continue
    bp = json.loads(bp_path.read_text())
    new = bp["params"]
    assert new["objective"] == "huber", model_id
    assert "alpha" in new, model_id

    s = path.read_text(encoding="utf-8")
    m = BLOCK_RE.search(s)
    assert m, model_id
    old_lines = m.group(2).splitlines()
    old_keys = []
    for line in old_lines:
        km = KEY_RE.match(line)
        assert km, (model_id, line)
        old_keys.append(km.group(1))
    keys = list(old_keys)
    if "alpha" not in keys:
        keys.insert(keys.index("objective") + 1, "alpha")
    # Only keys the module already declares (plus alpha) are written; the
    # tuner's fixed kwargs (subsample_freq, n_jobs, verbosity) and random_state
    # are kept at the module's existing values to leave its schema untouched.
    keep_existing = {"subsample_freq", "n_jobs", "verbosity", "random_state"}
    old_values = {}
    for line in old_lines:
        km = KEY_RE.match(line)
        old_values[km.group(1)] = km.group(2)
    out_lines = []
    for k in keys:
        if k in keep_existing:
            out_lines.append(f'    "{k}": {old_values[k]},')
        else:
            out_lines.append(f'    "{k}": {fmt(new[k])},')
    block = m.group(1) + "\n".join(out_lines) + m.group(3)
    s2 = s[: m.start()] + block + s[m.end():]

    # Comment above the params block: replace the tuning provenance line(s).
    cv_rmse = bp.get("cv_rmse_mean")
    cv = bp.get("cv_splits")
    src = (bp.get("alpha_derivation") or {}).get("scale_source", "mad")
    rule = "1.345 x 1.4826 x MAD" if src == "mad" else "1.345 x mean |y - median| (MAD is zero)"
    header = (
        f"# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv={cv}, RMSE scoring)\n"
        f"# on the full default predictor set; best mean cross-validated RMSE "
        f"{cv_rmse:.2f}. Huber threshold alpha = {rule} of the tuned target\n"
        f"# (2026-09-22 Huber retune, superseding the #169 MAE tune).\n"
    )
    # Replace the comment block immediately preceding the params assignment.
    s2 = re.sub(
        r"((?:^#.*\n)+)(_LGBM_(?:MAE|HUBER)_PARAMS: dict)",
        lambda mm: header + mm.group(2),
        s2,
        count=1,
        flags=re.MULTILINE,
    )
    for a, b in PROSE:
        s2 = s2.replace(a, b)
    if s2 != s:
        changed += 1
        if not DRY:
            path.write_text(s2, encoding="utf-8", newline="\n")
        else:
            print(f"would change {path.name}: n_est {old_values.get('n_estimators')} -> {new['n_estimators']}, alpha {new['alpha']:.3f}")
print(f"{'would change' if DRY else 'changed'} {changed} modules")
