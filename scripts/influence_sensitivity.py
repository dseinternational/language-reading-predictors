# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refit flagged single-period ITT/joint models without influential children.

Examples::

    python scripts/influence_sensitivity.py lrp-rli-itt-012 --config reporting
    python scripts/influence_sensitivity.py lrp-rli-itt-012 lrp-rli-itt-013 lrp-rli-itt-023 --config reporting

Each requested model must already have a completed, convergence-passing fit with
``pareto_k.csv`` under ``output/statistical_models/models/{model_id}-{config}``.
The command recomputes which child-level Pareto-k points exceed the saved ArviZ
threshold, excludes all flagged children in one direct leave-out refit, gates every free
variable, and writes a trace plus ``influence_sensitivity.csv`` centrally and
beside the original report. The displayed headline is the refit shift after both
posteriors are averaged over the same retained children; composition and total shifts
are shown separately.
"""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support

from language_reading_predictors import paths
from language_reading_predictors.statistical_models.influence import (
    run_influence_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="+",
        help="canonical registered ITT/joint model id(s)",
    )
    parser.add_argument(
        "--config",
        default="reporting",
        help="completed fit configuration to reproduce (default: reporting)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260715,
        help="random seed for each direct leave-out refit (default: 20260715)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="sampling cores (default: min of preset cores and saved chain count)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "override the output root (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR); the relative layout is unchanged"
        ),
    )
    args = parser.parse_args()

    paths.set_output_root(args.output_dir)
    print(f"Output root: {paths.describe_output_root()}")
    for model_id in args.models:
        print(f"\n=== {model_id}: direct influential-child sensitivity ===", flush=True)
        summary = run_influence_sensitivity(
            model_id,
            args.config,
            random_seed=args.random_seed,
            cores=args.cores,
        )
        display = summary[
            [
                "outcome",
                "refit_shift_ame_prob_median",
                "composition_shift_ame_prob_median",
                "total_shift_ame_prob_median",
                "ame_prob_median_full",
                "ame_prob_median_full_retained",
                "ame_prob_median_without_flagged",
                "prob_ame_pos_full",
                "prob_ame_pos_full_retained",
                "prob_ame_pos_without_flagged",
                "excluded_subject_ids",
                "converged",
            ]
        ].rename(
            columns={
                "refit_shift_ame_prob_median": "refit_shift_common_population",
                "composition_shift_ame_prob_median": "composition_shift",
                "total_shift_ame_prob_median": "total_shift",
                "ame_prob_median_full": "ame_full_sample",
                "ame_prob_median_full_retained": "ame_primary_retained",
                "ame_prob_median_without_flagged": "ame_leave_out_retained",
                "prob_ame_pos_full": "p_ame_positive_full",
                "prob_ame_pos_full_retained": "p_ame_positive_primary_retained",
                "prob_ame_pos_without_flagged": "p_ame_positive_leave_out",
            }
        )
        for column in (
            "refit_shift_common_population",
            "composition_shift",
            "total_shift",
            "ame_full_sample",
            "ame_primary_retained",
            "ame_leave_out_retained",
            "p_ame_positive_full",
            "p_ame_positive_primary_retained",
            "p_ame_positive_leave_out",
        ):
            display[column] = display[column].round(4)
        print(display.to_string(index=False))


if __name__ == "__main__":
    freeze_support()
    main()
