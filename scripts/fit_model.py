# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fits the specified model to the latest data. Saves plots and data to the output directory.
"""

import argparse
import os
import subprocess
from multiprocessing import freeze_support

from rich import print

from language_reading_predictors.models import model_lrp01, model_lrp02

if __name__ == "__main__":
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

    freeze_support()

    args = parser.parse_args()

    models = {
        "lrp01": model_lrp01,
        "lrp02": model_lrp02,
    }

    model_key = args.model.lower()

    if model_key == "all":
        contexts = [m.fit(args.config) for m in models.values()]
    elif model_key in models:
        contexts = [models[model_key].fit(args.config)]
    else:
        print(f"[bold red]Unknown model: {args.model}[/bold red]")
        print(f"Available models: {', '.join(models.keys())}")
        exit(1)

    # Summary table when fitting multiple models
    if len(contexts) > 1:
        print(
            "\n[green]============================================================[/green]"
        )
        print("[bold green]Summary[/bold green]")
        print(
            "[green]============================================================[/green]"
        )
        for ctx in contexts:
            cv_rmse = -ctx.cv_scores.mean() if ctx.cv_scores is not None else float("nan")
            print(
                f"  {ctx.config.model_id.upper():6s}  "
                f"target={ctx.config.target_var:20s}  "
                f"CV RMSE={cv_rmse:.4f}"
            )

    if args.render:
        for context in contexts:
            qmd_path = os.path.join(context.output_dir, "index.qmd")
            if os.path.exists(qmd_path):
                print(
                    f"\n[bold green]Rendering Quarto output: {qmd_path}[/bold green]"
                )
                subprocess.run(["quarto", "render", qmd_path], check=True)
            else:
                print(
                    f"\n[bold yellow]No index.qmd found at {qmd_path}, skipping render.[/bold yellow]"
                )
