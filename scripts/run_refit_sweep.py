# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resumable per-model driver for a full refit sweep (issue #554).

``scripts/fit_model.py all`` and ``scripts/fit_statistical_model.py all`` fit
every model in one process and batch all Quarto renders until after the last
fit, so an interrupted sweep leaves fitted-but-unrendered directories and no
per-model record of what was run. The August 2026 refit
(``notes/202608180929-full-statistical-refit-2026-08.md``) was therefore driven
by an untracked scratchpad script whose logic could not be reconstructed from
the artefacts, and whose skip rule checked completion markers only.

This driver is the checked-in replacement. It runs **one subprocess per model**
(fit and render together), streams that model's output to its own log file,
appends a JSON-lines journal record for every model, and — on resume — reuses a
stored fit only when its recorded identity still matches the current run:
sampling preset, source commit and dirty flag, the data file digest recorded in
the fit, and the environment lock digest.

Usage::

    python scripts/run_refit_sweep.py statistical --config reporting --render
    python scripts/run_refit_sweep.py gb --config reporting --render
    python scripts/run_refit_sweep.py statistical --config reporting --render \
        --models lrp-rli-itt-010,lrp-rli-itt-008 --force

Re-running the same command after an interruption continues where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from language_reading_predictors import paths  # noqa: E402


FIT_SCRIPTS = {
    "statistical": REPO_ROOT / "scripts" / "fit_statistical_model.py",
    "gb": REPO_ROOT / "scripts" / "fit_model.py",
}


def _sha256_file(path: Path) -> str | None:
    """Digest a file, returning ``None`` when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _git(arguments: list[str]) -> str | None:
    """Run one read-only Git query against this checkout."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


@dataclass(frozen=True)
class SweepIdentity:
    """The source/environment identity every reused fit must still match."""

    commit: str | None
    dirty: bool | None
    environment_sha256: str | None

    @classmethod
    def current(cls) -> SweepIdentity:
        status = _git(["status", "--porcelain"])
        return cls(
            commit=_git(["rev-parse", "HEAD"]),
            dirty=None if status is None else bool(status),
            environment_sha256=_sha256_file(REPO_ROOT / "environment.yml"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "dirty": self.dirty,
            "environment_sha256": self.environment_sha256,
        }


def _statistical_model_ids() -> list[str]:
    from language_reading_predictors.statistical_models.registry import discover_models

    return list(discover_models())


def _gb_model_ids(include_variants: bool) -> list[str]:
    from language_reading_predictors.models.registry import MODELS

    return [
        model_id
        for model_id, definition in MODELS.items()
        if include_variants or getattr(definition, "variant_of", None) is None
    ]


def _fit_dir(kind: str, model_id: str, config: str) -> Path:
    if kind == "statistical":
        return paths.stat_models_dir() / f"{model_id}-{config}"
    return paths.gb_models_dir() / model_id


def _reuse_reason(
    kind: str,
    directory: Path,
    config: str,
    identity: SweepIdentity,
    *,
    require_render: bool,
) -> str | None:
    """Return ``None`` when the stored fit is reusable, else why it is not.

    A completion marker alone is not enough: the August 2026 run record notes
    that a marker-only skip rule cannot tell a current fit from a stale one.
    """
    if not directory.is_dir():
        return "no stored fit"

    marker = "release_decision.json" if kind == "statistical" else "metrics.json"
    if not (directory / marker).is_file():
        return f"{marker} missing"
    if require_render and not (directory / "index.html").is_file():
        return "index.html missing"

    config_path = directory / "config.json"
    try:
        with open(config_path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "config.json unreadable"

    stored_config = stored.get("config_name") or stored.get("run_config")
    if stored_config != config:
        return f"sampling preset {stored_config!r} != {config!r}"

    if kind != "statistical":
        # GB fits record no source provenance, so preset plus markers is all the
        # identity there is; the fits are minutes long, so --force is cheap.
        return None

    source = (stored.get("provenance") or {}).get("source") or {}
    if source.get("commit") != identity.commit:
        return f"commit {source.get('commit')!r} != {identity.commit!r}"
    if bool(source.get("dirty")) != bool(identity.dirty):
        return f"dirty flag {source.get('dirty')!r} != {identity.dirty!r}"
    if stored.get("environment_lock_sha256") != identity.environment_sha256:
        return "environment.yml digest changed"

    data_path = stored.get("data_path")
    recorded_data_sha = stored.get("data_sha256")
    if data_path and recorded_data_sha:
        candidate = Path(data_path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if _sha256_file(candidate) != recorded_data_sha:
            return "data digest changed"

    return None


def _build_command(
    kind: str,
    model_id: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [sys.executable, str(FIT_SCRIPTS[kind]), model_id, "--config", args.config]
    if args.render:
        command.append("--render")
    if args.output_dir:
        command += ["--output-dir", args.output_dir]
    if kind == "statistical" and args.rli_randomised_archive:
        command += ["--rli-randomised-archive", args.rli_randomised_archive]
    if kind == "statistical" and args.target_accept is not None:
        command += ["--target-accept", str(args.target_accept)]
    return command


def _append_journal(journal_path: Path, record: dict[str, object]) -> None:
    with open(journal_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _run_one(
    kind: str,
    model_id: str,
    args: argparse.Namespace,
    *,
    log_path: Path,
) -> tuple[int, float]:
    """Fit (and optionally render) one model in its own process."""
    command = _build_command(kind, model_id, args)
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, time.monotonic() - started


def _selected_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        requested = [item.strip() for item in args.models.split(",") if item.strip()]
        return requested
    if args.kind == "statistical":
        return _statistical_model_ids()
    return _gb_model_ids(args.include_variants)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _print(message: str) -> None:
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=sorted(FIT_SCRIPTS), help="Which layer to sweep")
    parser.add_argument("--config", default="reporting", help="Sampling / run configuration")
    parser.add_argument("--render", action="store_true", help="Render each report right after its fit")
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: the whole registry)")
    parser.add_argument("--include-variants", action="store_true", help="GB only: include selection variants")
    parser.add_argument("--force", action="store_true", help="Refit even when a matching stored fit exists")
    parser.add_argument("--output-dir", default=None, help="Override the output root for this sweep")
    parser.add_argument("--rli-randomised-archive", default=None, help="Passed through to the statistical fits")
    parser.add_argument(
        "--target-accept",
        type=float,
        default=None,
        help=(
            "Override NUTS target_accept for the models in this run. Use for a "
            "remediation refit of a named model; never as a blanket escalation, "
            "which would LOWER acceptance on any module declaring a higher value."
        ),
    )
    parser.add_argument("--stop-on-failure", action="store_true", help="Abort the sweep at the first failure")
    parser.add_argument("--dry-run", action="store_true", help="List what would run, then exit")
    args = parser.parse_args(argv)

    paths.set_output_root(args.output_dir)
    identity = SweepIdentity.current()
    models = _selected_models(args)

    sweep_dir = paths.output_root() / "_sweep"
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    journal_path = sweep_dir / f"journal-{args.kind}-{args.config}.jsonl"

    _print(f"Output root:  {paths.output_root()}")
    _print(f"Sweep:        {args.kind} x {len(models)} models @ {args.config}"
           f"{' (+render)' if args.render else ''}")
    _print(f"Source:       {identity.commit} dirty={identity.dirty}")
    _print(f"Journal:      {journal_path}")

    plan: list[tuple[str, str | None]] = []
    for model_id in models:
        reason = _reuse_reason(
            args.kind,
            _fit_dir(args.kind, model_id, args.config),
            args.config,
            identity,
            require_render=args.render,
        )
        plan.append((model_id, reason))

    to_run = [model_id for model_id, reason in plan if args.force or reason is not None]
    reusable = [model_id for model_id, reason in plan if not args.force and reason is None]
    _print(f"Reusing {len(reusable)} stored fit(s); running {len(to_run)}.")

    if args.dry_run:
        for model_id, reason in plan:
            state = "RUN " if (args.force or reason is not None) else "skip"
            _print(f"  {state} {model_id}" + (f"  ({reason})" if reason else ""))
        return 0

    failures: list[str] = []
    sweep_started = time.monotonic()
    for index, model_id in enumerate(to_run, start=1):
        log_path = log_dir / f"{args.kind}-{model_id}-{args.config}.log"
        _print(f"[{index}/{len(to_run)}] {model_id} ...")
        started_at = datetime.now(UTC).isoformat()
        returncode, seconds = _run_one(args.kind, model_id, args, log_path=log_path)
        status = "ok" if returncode == 0 else "failed"
        if returncode != 0:
            failures.append(model_id)
        _append_journal(
            journal_path,
            {
                "model_id": model_id,
                "kind": args.kind,
                "config": args.config,
                "rendered": bool(args.render),
                "status": status,
                "returncode": returncode,
                "seconds": round(seconds, 3),
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "log": str(log_path),
                "identity": identity.as_dict(),
                "driver_sha256": _sha256_file(Path(__file__)),
            },
        )
        _print(
            f"[{index}/{len(to_run)}] {model_id} {status} in {_format_duration(seconds)}"
            + ("" if returncode == 0 else f" (exit {returncode}; see {log_path})")
        )
        if returncode != 0 and args.stop_on_failure:
            break

    _print("")
    _print(f"Sweep finished in {_format_duration(time.monotonic() - sweep_started)}: "
           f"{len(to_run) - len(failures)} ok, {len(failures)} failed.")
    if failures:
        _print("Failed: " + ", ".join(failures))
        return 1
    return 0


def _iter_journal(path: Path) -> Iterable[dict[str, object]]:
    """Read a sweep journal back (used by ad-hoc reporting over a finished run)."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
