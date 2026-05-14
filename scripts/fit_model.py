# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fits the specified model to the latest data. Saves plots and data to the output directory.
"""

import argparse
import subprocess
from pathlib import Path
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
from rich import print

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
)
from language_reading_predictors.models.registry import MODELS


def _fit(cfg, config_name):
    from language_reading_predictors.models.common import RunConfig

    return cfg.pipeline_cls(cfg, RunConfig.from_name(config_name)).fit()


def main():
    parser = argparse.ArgumentParser(
        description="Fit a language-reading-predictors model."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model id (e.g. LRP01) or 'all'.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Quarto model output after fitting.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dev",
        help="Run configuration: dev (fast), test (moderate), reporting (full). Default: dev.",
    )
    parser.add_argument(
        "--include-variants",
        action="store_true",
        help=(
            "When model is 'all', also fit selection variants (entries where "
            "variant_of is set). Ignored for explicit model ids."
        ),
    )

    freeze_support()

    setup.init_script()

    args = parser.parse_args()

    model_key = args.model.lower()

    if model_key == "all":
        models_to_run = [
            cfg
            for cfg in MODELS.values()
            if args.include_variants or cfg.variant_of is None
        ]
        if not models_to_run:
            print("[bold red]No models to run.[/bold red]")
            raise SystemExit(1)
    elif model_key in MODELS:
        models_to_run = [MODELS[model_key]]
    else:
        print(f"[bold red]Unknown model: {args.model}[/bold red]")
        print(f"Available models: {', '.join(MODELS.keys())}")
        raise SystemExit(1)

    contexts = []
    failed = []
    for cfg in models_to_run:
        try:
            contexts.append(_fit(cfg, args.config))
        except Exception as exc:
            failed.append((cfg.model_id, exc))
            print(f"[bold red]Error fitting {cfg.model_id}: {exc}[/bold red]")

    # Summary table when fitting multiple models
    if len(models_to_run) > 1:
        rows = []
        for ctx in contexts:
            cv_df = ctx.dataframes.get("cv_scores")
            rows.append(
                {
                    "model": ctx.config.model_id.upper(),
                    "target": ctx.config.target_var,
                    "n_obs": int(len(ctx.X)) if ctx.X is not None else None,
                    "cv_rmse": float(cv_df["rmse"].mean()) if cv_df is not None else None,
                    "cv_rmse_std": float(cv_df["rmse"].std()) if cv_df is not None else None,
                    "cv_mae": float(cv_df["mae"].mean()) if cv_df is not None else None,
                    "cv_r2": float(cv_df["r2"].mean()) if cv_df is not None else None,
                    "status": "ok",
                }
            )
        for model_id, exc in failed:
            rows.append(
                {
                    "model": model_id.upper(),
                    "target": "—",
                    "n_obs": None,
                    "cv_rmse": None,
                    "cv_rmse_std": None,
                    "cv_mae": None,
                    "cv_r2": None,
                    "status": f"FAILED: {type(exc).__name__}",
                }
            )

        print()
        print_table(
            metrics_table(
                rows,
                title=f"Summary ({len(contexts)} fitted, {len(failed)} failed)",
                columns=[
                    "model",
                    "target",
                    "n_obs",
                    "cv_rmse",
                    "cv_rmse_std",
                    "cv_mae",
                    "cv_r2",
                    "status",
                ],
            )
        )

    render_failed: list[tuple[str, subprocess.CalledProcessError]] = []
    if args.render:
        for context in contexts:
            qmd_path = Path(context.output_dir) / "index.qmd"
            if not qmd_path.exists():
                print(
                    f"\n[bold yellow]No index.qmd found at {qmd_path}, skipping render.[/bold yellow]"
                )
                continue
            print(f"\n[bold green]Rendering Quarto output: {qmd_path}[/bold green]")
            try:
                subprocess.run(["quarto", "render", str(qmd_path)], check=True)
            except subprocess.CalledProcessError as exc:
                # Don't abort the loop on one bad render — the user will
                # want the other reports built, and we'll surface a
                # non-zero exit at the end.
                render_failed.append((context.config.model_id, exc))
                print(
                    f"[bold red]Render failed for {context.config.model_id}: "
                    f"quarto exited {exc.returncode}[/bold red]"
                )

    if failed or render_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
