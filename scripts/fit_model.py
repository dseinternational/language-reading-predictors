# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fits the specified model to the latest data. Saves plots and data to the output directory.
"""

import argparse
from multiprocessing import freeze_support

from rich import print

from language_reading_predictors.models import model_lrp01

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a language-reading-predictors model."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model id (e.g. LRP01) or 'all'.",
    )

    freeze_support()

    args = parser.parse_args()

    models = {
        "lrp01": model_lrp01,
    }

    model_key = args.model.lower()

    if model_key == "all":
        for m in models.values():
            m.fit()
    elif model_key in models:
        models[model_key].fit()
    else:
        print(f"[bold red]Unknown model: {args.model}[/bold red]")
        print(f"Available models: {', '.join(models.keys())}")
        exit(1)
