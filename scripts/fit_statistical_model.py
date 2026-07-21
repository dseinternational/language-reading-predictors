# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI entry point for fitting the Bayesian statistical models.

Usage::

    python scripts/fit_statistical_model.py lrp-rli-itt-007 --config dev
    python scripts/fit_statistical_model.py all --config dev
    python scripts/fit_statistical_model.py lrp-rli-itt-010 --config reporting --render
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
import uuid
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
from rich import print as rprint
from rich.panel import Panel

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_panel,
    print_table,
)
from language_reading_predictors import model_ids
from language_reading_predictors import paths
from language_reading_predictors.storage import upload_to_blob_storage
from language_reading_predictors.statistical_models.registry import (
    discover_models,
)
from language_reading_predictors.statistical_models.run_options import (
    StatisticalRunOptions,
)
from language_reading_predictors.statistical_models.provenance import (
    write_failure_record,
)


# Auto-discovered (#165): every statistical_models submodule that defines its
# own top-level fit() is registered under its module name (== its CLI id), so a
# new lrp.../rlm... model needs no edit here. See statistical_models/registry.py.
MODELS = discover_models()


def main() -> None:
    # Apply the shared DSE matplotlib house style so Bayesian figures match the
    # GB ones (scripts/fit_model.py does the same). Without this every figure
    # falls back to matplotlib/ArviZ defaults and looks inconsistent in the report.
    setup.init_script()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help=(
            "Model id (for example lrp-rli-itt-001, lrp-rli-did-001, lrp-rli-gf-001, lrp-rli-lf-001, "
            "lrp-rli-al-001, lrp-rli-dose-077, lrp-rli-adj-065, lrp-rli-lcsm-067) or 'all'"
        ),
    )
    parser.add_argument(
        "--config",
        default="dev",
        help=(
            "Sampling configuration: dev, test, rep-lite, reporting "
            "(rep-lite is reporting-grade rigour but lighter — see dse_research_utils.sampling)"
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Quarto report after fitting",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=None,
        help="Override NUTS target_accept (default: preset from --config)",
    )
    parser.add_argument(
        "--reuse-trace",
        action="store_true",
        help=(
            "Re-emit artefacts from each model's existing trace.nc instead of "
            "sampling: rebuilds the model, loads the saved posterior, recomputes "
            "LOO / posterior-predictive, and redraws every figure. Use to backfill "
            "figure or template changes onto already-fitted models without NUTS."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload model output to Azure Blob Storage after fitting.",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include trace files (.nc) in the upload (excluded by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR); the relative layout is unchanged. Default: "
            "repo-local output/."
        ),
    )
    args = parser.parse_args()

    paths.set_output_root(args.output_dir)
    rprint(f"[bold]Output root:[/bold] {paths.describe_output_root()}")

    if args.reuse_trace:
        os.environ["DSE_LRP_REUSE_TRACE"] = "1"
        rprint(
            "[cyan]Reuse-trace mode: loading each model's saved posterior instead "
            "of sampling; artefacts are re-emitted from the existing draws.[/cyan]"
        )

    run_options = StatisticalRunOptions(target_accept=args.target_accept)
    if run_options.target_accept is not None:
        rprint(
            f"[yellow]Overriding target_accept -> {run_options.target_accept}[/yellow]"
        )

    # The registry is keyed on the canonical CLI id (``lrp-rli-itt-001``) since #168
    # Phase 2. Build a legacy-alias index over those keys so a legacy id
    # (``lrpitt01``) or any canonical form/case still resolves — each canonical key
    # maps back to its legacy/display/module forms without needing a ``kind`` (the
    # family is embedded in the canonical id).
    aliases: dict[str, str] = {}
    for key in MODELS:
        aliases[key.lower()] = key
        try:
            mid = model_ids.parse_canonical(key)
        except model_ids.ModelIdError:
            continue
        for form in (mid.legacy, mid.display, mid.module):
            aliases[form.lower()] = key
    normalised = args.model.strip().lower()
    requested = aliases.get(normalised, normalised)
    if requested == "all":
        to_fit = list(MODELS.items())
    elif requested in MODELS:
        to_fit = [(requested, MODELS[requested])]
    else:
        rprint(f"[red]Unknown model: {args.model}[/red]")
        rprint(f"[yellow]Available: {', '.join(MODELS)}[/yellow]")
        raise SystemExit(1)

    rprint()
    print_panel(
        Panel(
            "[bold]Statistical model run[/bold]\n\n"
            f"[dim]Models:[/dim]     {', '.join(mid.upper() for mid, _ in to_fit)}\n"
            f"[dim]Run config:[/dim] {args.config}"
            + (
                f"\n[dim]Override:[/dim]   target_accept={args.target_accept}"
                if args.target_accept is not None
                else ""
            ),
            border_style="green",
            padding=(1, 2),
        )
    )

    contexts: list = []
    failed: list[tuple[str, Exception]] = []
    for model_id, module in to_fit:
        try:
            contexts.append(module.fit(args.config, options=run_options))
        except Exception as exc:
            failed.append((model_id, exc))
            traceback_text = traceback.format_exc()
            rprint(f"[bold red]Error fitting {model_id}: {exc}[/bold red]")
            print(traceback_text, file=sys.stderr, end="")
            try:
                failure_path = write_failure_record(
                    paths.output_root(),
                    model_id=model_id,
                    config=args.config,
                    error=exc,
                    traceback_text=traceback_text,
                )
            except (OSError, TypeError, ValueError) as record_error:
                rprint(
                    "[bold yellow]Could not save the structured failure record: "
                    f"{record_error}[/bold yellow]"
                )
            else:
                rprint(f"[dim]Failure record: {failure_path}[/dim]")

    if len(to_fit) > 1 or failed:
        rows = []
        for ctx in contexts:
            spec = ctx.spec
            prepared = ctx.prepared
            rows.append(
                {
                    "model": spec.model_id.upper(),
                    "kind": spec.kind,
                    "outcome": spec.outcome_symbol or "-",
                    "mechanism": spec.mechanism_symbol or "-",
                    "n_obs": int(prepared.n_obs) if prepared is not None else None,
                    "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
                    "status": "ok",
                }
            )
        for model_id, exc in failed:
            rows.append(
                {
                    "model": model_id.upper(),
                    "kind": "-",
                    "outcome": "-",
                    "mechanism": "-",
                    "n_obs": None,
                    "loo_elpd": None,
                    "status": f"FAILED: {type(exc).__name__}",
                }
            )
        rprint()
        print_table(
            metrics_table(
                rows,
                title=f"Summary ({len(contexts)} fitted, {len(failed)} failed)",
                columns=["model", "kind", "outcome", "mechanism", "n_obs", "loo_elpd", "status"],
            )
        )

    render_failed: list[tuple[str, subprocess.CalledProcessError]] = []
    if args.render:
        for ctx in contexts:
            qmd = os.path.join(ctx.output_dir, "index.qmd")
            if not os.path.exists(qmd):
                rprint(f"[bold yellow]No index.qmd at {qmd}, skipping render.[/bold yellow]")
                continue
            rprint(f"[bold green]quarto render {qmd}[/bold green]")
            try:
                # Pin Quarto's Jupyter kernel to the interpreter that ran the
                # fit (it has arviz/pymc/etc.). Otherwise Quarto resolves its
                # own ``python3`` from PATH, which fails when the conda env is
                # not the first interpreter on PATH.
                render_env = {**os.environ, "QUARTO_PYTHON": sys.executable}
                subprocess.run(["quarto", "render", qmd], check=True, env=render_env)
            except subprocess.CalledProcessError as exc:
                # Don't abort the loop on one bad render — build the rest and
                # surface a non-zero exit at the end (mirrors fit_model.py).
                label = os.path.basename(str(ctx.output_dir))
                render_failed.append((label, exc))
                rprint(
                    f"[bold red]Render failed for {label}: "
                    f"quarto exited {exc.returncode}[/bold red]"
                )

    if args.upload:
        run_id = str(uuid.uuid7())
        for ctx in contexts:
            upload_to_blob_storage(
                str(ctx.output_dir),
                os.path.basename(str(ctx.output_dir)),
                include_traces=args.include_traces,
                run_id=run_id,
            )

    if failed or render_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    freeze_support()
    main()
