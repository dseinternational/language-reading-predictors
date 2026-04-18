# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI entry point for fitting the Bayesian statistical models (LRP52-LRP58).

Usage::

    python scripts/fit_statistical_model.py lrp52 --config dev
    python scripts/fit_statistical_model.py all --config dev
    python scripts/fit_statistical_model.py lrp52 --config reporting --render
"""

from __future__ import annotations

import argparse
import os
import subprocess
from multiprocessing import freeze_support

from rich import print as rprint

from language_reading_predictors.statistical_models import (
    lrp52,
    lrp53,
    lrp54,
    lrp55,
    lrp56,
    lrp57,
    lrp58,
)


MODELS = {
    "lrp52": lrp52,
    "lrp53": lrp53,
    "lrp54": lrp54,
    "lrp55": lrp55,
    "lrp56": lrp56,
    "lrp57": lrp57,
    "lrp58": lrp58,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model id (lrp52..lrp58) or 'all'")
    parser.add_argument(
        "--config",
        default="dev",
        help="Sampling configuration: dev, test, reporting (see dse_research_utils.sampling)",
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
    args = parser.parse_args()

    if args.target_accept is not None:
        import dse_research_utils.statistics.models.sampling as _S

        _orig = _S.get_sampling_configuration

        def _override(cfg: str = "dev", random_seed: int = 47):
            s = _orig(cfg, random_seed=random_seed)
            s.target_accept = args.target_accept
            return s

        _S.get_sampling_configuration = _override
        rprint(
            f"[yellow]Overriding target_accept -> {args.target_accept}[/yellow]"
        )

    if args.model == "all":
        to_fit = list(MODELS.values())
    elif args.model in MODELS:
        to_fit = [MODELS[args.model]]
    else:
        rprint(f"[red]Unknown model: {args.model}[/red]")
        rprint(f"[yellow]Available: {', '.join(MODELS)}[/yellow]")
        raise SystemExit(1)

    contexts = [m.fit(args.config) for m in to_fit]

    if args.render:
        for ctx in contexts:
            qmd = os.path.join(ctx.output_dir, "index.qmd")
            if os.path.exists(qmd):
                rprint(f"[bold green]quarto render {qmd}[/bold green]")
                subprocess.run(["quarto", "render", qmd], check=False)


if __name__ == "__main__":
    freeze_support()
    main()
