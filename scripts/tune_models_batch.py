# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Batch orchestrator for the GB-model hyperparameter retune (issue #169).

``scripts/tune_model.py`` tunes one model at a time. This wraps it so the full
50-model retune can be launched, resumed, and audited as a single job. Each
model is tuned in its **own subprocess** (clean memory, isolated failure), using
the reviewed tuning policy from issue #169::

    python scripts/tune_model.py {model_id} --n-trials 150 --scoring mae --lgbm-objective mae

Design decisions
----------------
* **Sequential.** Each Optuna trial already saturates all cores via LightGBM's
  ``n_jobs=-1``; running models concurrently oversubscribes threads and is
  slower overall. So models run one at a time.
* **Resumable.** A model whose ``best_params.json`` exists (and matches the
  requested policy) is treated as complete and skipped unless ``--force``.
* **Auditable.** A JSON manifest under ``output/tuning/`` records, per model, the
  command args, git commit, start/end timestamps, wall-clock, status, and output
  path. It is rewritten after every model so a killed run leaves a truthful
  record and can be resumed.
* **Fault-tolerant.** A model that fails is recorded and the batch continues; a
  final summary lists every failure.

Families (``--family``)
-----------------------
* ``gain``        — ``lrp-rli-gbg-001`` … ``lrp-rli-gbg-022`` (22 models)
* ``level``       — ``lrp-rli-gbl-001`` … ``lrp-rli-gbl-028`` (28 models)
* ``core``        — outcomes ``001``–``016`` (gain + level)
* ``exploratory`` — outcomes ``017``–``028`` (gain + level)
* ``all``         — every registered GB model (default)

Usage
-----
    python scripts/tune_models_batch.py --dry-run
    python scripts/tune_models_batch.py --family gain
    python scripts/tune_models_batch.py --models lrp-rli-gbg-012 lrp-rli-gbl-012
    python scripts/tune_models_batch.py --force --n-trials 150
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from language_reading_predictors import paths as _paths
from language_reading_predictors.models.registry import MODELS

# Canonical GB model ids since #168 Phase 2, e.g. ``lrp-rli-gbg-012`` /
# ``lrp-rli-gbl-028`` (MODELS is keyed on the canonical CLI id).
_GB_RE = re.compile(r"^lrp-rli-gb([gl])-(\d+)$")

MANIFEST_NAME = "retune169_manifest.json"


def _all_gb_models() -> list[str]:
    """Registered GB models, sorted gain-before-level then by outcome number."""
    matched = [(m.group(1), int(m.group(2)), k) for k in MODELS if (m := _GB_RE.match(k))]
    matched.sort(key=lambda t: (t[0], t[1]))
    return [k for _, _, k in matched]


def _select_models(family: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        unknown = [m for m in explicit if m not in MODELS]
        if unknown:
            raise SystemExit(f"Unknown model id(s): {', '.join(unknown)}")
        return explicit
    gb = _all_gb_models()
    if family == "all":
        return gb
    if family == "gain":
        return [k for k in gb if _GB_RE.match(k).group(1) == "g"]
    if family == "level":
        return [k for k in gb if _GB_RE.match(k).group(1) == "l"]
    if family in ("core", "exploratory"):
        out = []
        for k in gb:
            n = int(_GB_RE.match(k).group(2))
            if family == "core" and n <= 16:
                out.append(k)
            elif family == "exploratory" and n >= 17:
                out.append(k)
        return out
    raise SystemExit(f"Unknown family: {family}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_paths.ROOT_DIR, text=True
        ).strip()
    except Exception:  # noqa: BLE001 — best-effort provenance
        return "unknown"


def _is_complete(model_id: str, scoring: str, objective: str) -> bool:
    """A model is complete if its best_params.json exists and matches the policy."""
    bp = _paths.gb_tuning_dir() / model_id / "best_params.json"
    if not bp.is_file():
        return False
    try:
        data = json.loads(bp.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    # Only treat as complete if it was tuned under the SAME policy we'd re-run,
    # so a leftover RMSE tune doesn't masquerade as a done MAE retune.
    if data.get("scoring") != scoring:
        return False
    if data.get("params", {}).get("objective") != objective:
        return False
    return True


def _tune_command(model_id: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(_paths.ROOT_DIR / "scripts" / "tune_model.py"),
        model_id,
        "--n-trials",
        str(args.n_trials),
        "--scoring",
        args.scoring,
        "--lgbm-objective",
        args.lgbm_objective,
        "--seed",
        str(args.seed),
    ]
    if args.timeout is not None:
        cmd += ["--timeout", str(args.timeout)]
    if args.output_dir is not None:
        cmd += ["--output-dir", args.output_dir]
    return cmd


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch GB-model retune (issue #169).")
    parser.add_argument(
        "--family",
        choices=["all", "gain", "level", "core", "exploratory"],
        default="all",
        help="Which GB models to tune (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit model ids (overrides --family).",
    )
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--scoring", default="mae", choices=["rmse", "mae", "medae"])
    parser.add_argument("--lgbm-objective", default="mae")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--timeout", type=float, default=None, help="Per-model study timeout (s).")
    parser.add_argument(
        "--force", action="store_true", help="Re-tune even models already complete."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List planned models and commands; do nothing."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    _paths.set_output_root(args.output_dir)

    models = _select_models(args.family, args.models)
    manifest_path = _paths.gb_tuning_dir() / MANIFEST_NAME
    logs_dir = _paths.gb_tuning_dir() / "_logs"

    commit = _git_commit()
    print(f"Output root : {_paths.describe_output_root()}")
    print(f"Git commit  : {commit}")
    print(f"Policy      : --n-trials {args.n_trials} --scoring {args.scoring} "
          f"--lgbm-objective {args.lgbm_objective} --seed {args.seed}")
    print(f"Manifest    : {manifest_path}")
    print(f"Models ({len(models)}): {', '.join(models)}")
    print()

    if args.dry_run:
        print("DRY RUN — planned actions:")
        for m in models:
            done = _is_complete(m, args.scoring, args.lgbm_objective)
            action = "SKIP (complete)" if (done and not args.force) else "TUNE"
            print(f"  [{action:15s}] {m}")
            if action == "TUNE":
                print(f"      {' '.join(_tune_command(m, args))}")
        return

    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "issue": 169,
        "policy": {
            "n_trials": args.n_trials,
            "scoring": args.scoring,
            "lgbm_objective": args.lgbm_objective,
            "seed": args.seed,
            "timeout": args.timeout,
        },
        "git_commit": commit,
        "batch_started": _utc_now(),
        "batch_finished": None,
        "models": {},
    }
    # Preserve prior manifest entries (resume): keep records for models not in this run.
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text())
            manifest["models"] = prior.get("models", {})
        except (json.JSONDecodeError, OSError):
            pass

    failures: list[str] = []
    skipped: list[str] = []
    succeeded: list[str] = []

    for i, model_id in enumerate(models, 1):
        if not args.force and _is_complete(model_id, args.scoring, args.lgbm_objective):
            print(f"[{i}/{len(models)}] {model_id}: SKIP (already complete)")
            skipped.append(model_id)
            manifest["models"].setdefault(model_id, {})["status"] = "skipped_complete"
            _write_manifest(manifest_path, manifest)
            continue

        cmd = _tune_command(model_id, args)
        log_path = logs_dir / f"{model_id}.log"
        start = _utc_now()
        print(f"[{i}/{len(models)}] {model_id}: tuning… (log: {log_path})", flush=True)

        entry = {
            "status": "running",
            "command": cmd,
            "git_commit": commit,
            "start": start,
            "end": None,
            "wall_seconds": None,
            "log": str(log_path),
            "output": str(_paths.gb_tuning_dir() / model_id),
        }
        manifest["models"][model_id] = entry
        _write_manifest(manifest_path, manifest)

        t0 = datetime.now(timezone.utc)
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=_paths.ROOT_DIR)
        wall = (datetime.now(timezone.utc) - t0).total_seconds()

        entry["end"] = _utc_now()
        entry["wall_seconds"] = round(wall, 1)
        entry["returncode"] = proc.returncode
        if proc.returncode == 0:
            entry["status"] = "success"
            succeeded.append(model_id)
            # Surface headline metric into the manifest for quick review.
            try:
                bp = json.loads((_paths.gb_tuning_dir() / model_id / "best_params.json").read_text())
                entry["cv_mean"] = bp.get(f"cv_{args.scoring}_mean")
                entry["cv_std"] = bp.get(f"cv_{args.scoring}_std")
                entry["n_estimators"] = bp.get("params", {}).get("n_estimators")
            except (json.JSONDecodeError, OSError, KeyError):
                pass
            print(f"      done in {wall / 60:.1f} min  "
                  f"(CV {args.scoring}={entry.get('cv_mean')}, n_est={entry.get('n_estimators')})")
        else:
            entry["status"] = "failed"
            failures.append(model_id)
            print(f"      FAILED (returncode {proc.returncode}) after {wall / 60:.1f} min "
                  f"— see {log_path}")
        _write_manifest(manifest_path, manifest)

    manifest["batch_finished"] = _utc_now()
    _write_manifest(manifest_path, manifest)

    print()
    print("=" * 60)
    print(f"Batch complete: {len(succeeded)} tuned, {len(skipped)} skipped, "
          f"{len(failures)} failed.")
    if failures:
        print(f"Failures: {', '.join(failures)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
