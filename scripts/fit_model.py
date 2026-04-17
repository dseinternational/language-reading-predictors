# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fits the specified model to the latest data. Saves plots and data to the output directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing import freeze_support

# Windows consoles default to cp1252 and crash on non-ASCII chars via
# rich. Prefer UTF-8 so model descriptions can use arrows / en-dashes /
# accented characters without breaking the run.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import dse_research_utils.environment.setup as setup
from rich import print
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
        print(
            "\n[green]============================================================[/green]"
        )
        print("[bold green]Summary[/bold green]")
        print(
            "[green]============================================================[/green]"
        )
        for ctx in contexts:
            cv_scores = getattr(ctx, "cv_scores", None)
            if cv_scores is not None:
                summary = f"CV RMSE={cv_scores.mean():.4f}"
            elif getattr(ctx, "trace", None) is not None:
                # Bayesian pipeline: no CV. Surface divergence count as the
                # at-a-glance health signal.
                ss = ctx.trace.sample_stats
                n_div = (
                    int(ss.get("diverging").sum().item())
                    if "diverging" in ss
                    else 0
                )
                summary = f"divergences={n_div}"
            else:
                summary = "(no metrics)"
            print(
                f"  {ctx.config.model_id.upper():6s}  "
                f"target={ctx.config.target_var:20s}  "
                f"{summary}"
            )
        for model_id, exc in failed:
            print(f"  [red]{model_id.upper():6s}  FAILED: {exc}[/red]")

    if args.render:
        for context in contexts:
            qmd_path = Path(context.output_dir) / "index.qmd"
            if qmd_path.exists():
                print(f"\n[bold green]Rendering Quarto output: {qmd_path}[/bold green]")
                subprocess.run(["quarto", "render", str(qmd_path)], check=True)
            else:
                print(
                    f"\n[bold yellow]No index.qmd found at {qmd_path}, skipping render.[/bold yellow]"
                )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
