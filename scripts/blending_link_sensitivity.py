# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the mandatory trace-backed phoneme-blending link comparison.

Run after both ``lrp-rli-itt-008`` and ``lrp-rli-itt-108`` have completed at the
same configuration, and before regenerating key findings or rendering reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.blending_sensitivity import (
    BLENDING_LINK_MODELS,
    BLENDING_SENSITIVITY_FILENAME,
    build_blending_link_sensitivity,
)
from language_reading_predictors.statistical_models.reporting import (
    generate_key_findings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="reporting",
        choices=("dev", "test", "rep-lite", "reporting"),
        help="Completed-fit configuration to compare (default: reporting).",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help=(
            "Central content-addressed archive. Defaults to "
            "<statistical-model-root>/blending_link_sensitivity."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    models = Path(_paths.stat_models_dir())
    statistical_root = models.parent
    archive = args.archive_dir or statistical_root / "blending_link_sensitivity"
    summary = build_blending_link_sensitivity(
        models,
        archive,
        config_name=args.config,
        install_report_copies=True,
    )
    print(
        f"Validated {len(summary)} B link fits; wrote "
        f"{archive / BLENDING_SENSITIVITY_FILENAME} and installed report copies."
    )
    for model_id, _link in BLENDING_LINK_MODELS:
        output_dir = models / f"{model_id}-{args.config}"
        payload = generate_key_findings(output_dir)
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"{model_id} key findings were not regenerated cleanly: {payload}"
            )
    print(
        "Regenerated trace-bound key findings for lrp-rli-itt-008 and "
        "lrp-rli-itt-108. Next: render both reports."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
