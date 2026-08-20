# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare the step-1 gradient-boosting rankings with the step-2 coefficients.

``scripts/compare_horseshoe_vs_gb.py`` already checks the gradient-boosting
ranking against the Bayesian *many-predictor* ranking (`horseshoe`). Issue #554
item 5 also asks for the comparison against the families that estimate the same
associations structurally:

* **gains** — ``gain_factors`` (`gf-001`), whose ``gamma_*`` coefficients are the
  adjusted associations of each period-start predictor with the period's
  post-score, against the GB model of word-reading *change* (`gbg-012`);
* **levels** — ``pooled_levels`` (`pl-001`, `pl-003`–`006`), whose between-child
  coefficients are the adjusted associations with word-reading *level*, against
  the GB model of word-reading *level* (`gbl-012`).

Both sides are reduced to construct symbols and compared by rank. The point is
not to validate either layer against the other — they estimate different things
under different assumptions — but to see whether the ordering a tree ensemble
finds survives a model that adjusts deliberately.

Usage::

    python scripts/compare_gb_vs_statistical.py

Writes ``gb_vs_statistical.csv`` to ``output/statistical_models/comparison/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_horseshoe_vs_gb import (  # noqa: E402
    column_to_symbol_map,
    gb_construct_ranking,
)

from language_reading_predictors import paths  # noqa: E402

# gain_factors coefficient -> construct symbol. The treatment term is excluded:
# it is the one causal coefficient and has no counterpart in a GB ranking.
GF_TERM_TO_SYMBOL = {
    "gamma_own": "W",
    "gamma_A": "age",
    "gamma_ability": "blocks",
    "gamma_L": "L",
    "gamma_B": "B",
    "gamma_N": "N",
    "gamma_R": "R",
    "gamma_E": "E",
    "gamma_TR": "TR",
    "gamma_TE": "TE",
}

# Two pooled_levels exposures are composites whose components are separate GB
# columns, so they need adding to the shared construct map: speech production is
# the sum of the three DEAP picture-naming scores, phonological memory the two
# repetition scores. ``gb_construct_ranking`` takes the max importance among a
# construct's columns, which is the right aggregation for a composite too.
EXTRA_COLUMN_SYMBOLS = {
    "deappin": "deapp_c",
    "deappvo": "deapp_c",
    "deappfi": "deapp_c",
    "deappav": "deapp_c",
    "erbnw": "erbto",
    "erbword": "erbto",
}


# pooled_levels model -> the construct its exposure measures.
PL_MODEL_TO_SYMBOL = {
    "lrp-rli-pl-001": "L",
    "lrp-rli-pl-003": "E",
    "lrp-rli-pl-004": "R",
    "lrp-rli-pl-005": "erbto",
    "lrp-rli-pl-006": "deapp_c",
}


def _rank(values: pd.Series) -> pd.Series:
    """Rank with 1 = most important, average-rank ties."""
    return values.rank(ascending=False, method="average")


def gain_comparison(config: str) -> pd.DataFrame:
    """GB word-reading change against the gain-factor adjusted associations."""
    stat_dir = paths.stat_models_dir() / f"lrp-rli-gf-001-{config}"
    factor = pd.read_csv(stat_dir / "factor_summary.csv")
    stats = []
    for _, row in factor.iterrows():
        symbol = GF_TERM_TO_SYMBOL.get(str(row["term"]))
        if symbol is None:
            continue
        stats.append(
            {
                "symbol": symbol,
                "stat_median": float(row["median"]),
                "stat_abs": abs(float(row["median"])),
                "stat_prob": float(row["prob_positive"]),
            }
        )
    stat_frame = pd.DataFrame(stats)

    gb = pd.read_csv(paths.gb_models_dir() / "lrp-rli-gbg-012" / "predictor_ranking.csv")
    gb_frame = gb_construct_ranking(gb, column_to_symbol_map())
    merged = stat_frame.merge(gb_frame, on="symbol", how="inner")
    merged["comparison"] = "gain: gbg-012 vs gf-001"
    return merged


def level_comparison(config: str) -> pd.DataFrame:
    """GB word-reading level against the pooled between-child associations."""
    stats = []
    for model_id, symbol in PL_MODEL_TO_SYMBOL.items():
        path = paths.stat_models_dir() / f"{model_id}-{config}" / "pooled_levels_summary.csv"
        if not path.is_file():
            continue
        table = pd.read_csv(path).set_index("term")
        if "beta_between" not in table.index:
            continue
        row = table.loc["beta_between"]
        stats.append(
            {
                "symbol": symbol,
                "stat_median": float(row["median"]),
                "stat_abs": abs(float(row["median"])),
                "stat_prob": float(row["prob_positive"]),
            }
        )
    stat_frame = pd.DataFrame(stats)

    gb = pd.read_csv(paths.gb_models_dir() / "lrp-rli-gbl-012" / "predictor_ranking.csv")
    gb_frame = gb_construct_ranking(gb, {**column_to_symbol_map(), **EXTRA_COLUMN_SYMBOLS})
    merged = stat_frame.merge(gb_frame, on="symbol", how="inner")
    merged["comparison"] = "level: gbl-012 vs pooled_levels between-child"
    return merged


def summarise(frame: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import spearmanr

    if len(frame) < 3:
        return {"n_shared": int(len(frame)), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    result = spearmanr(frame["stat_abs"], frame["gb_perm_imp"])
    return {
        "n_shared": int(len(frame)),
        "spearman_rho": float(result.statistic),
        "spearman_p": float(result.pvalue),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="reporting")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths.set_output_root(args.output_dir)

    frames = []
    for builder in (gain_comparison, level_comparison):
        frame = builder(args.config)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["stat_rank"] = _rank(frame["stat_abs"])
        frame["gb_rank_shared"] = _rank(frame["gb_perm_imp"])
        stats = summarise(frame)
        frame["spearman_rho"] = stats["spearman_rho"]
        frame["n_shared"] = stats["n_shared"]
        frames.append(frame)
        print(f"\n=== {frame['comparison'].iloc[0]}")
        print(f"    shared constructs: {stats['n_shared']}  Spearman rho = {stats['spearman_rho']:+.2f}")
        show = frame.sort_values("stat_rank")[
            ["symbol", "stat_median", "stat_rank", "gb_perm_imp", "gb_rank_shared"]
        ]
        print(show.to_string(index=False))

    if not frames:
        print("no comparison could be built", file=sys.stderr)
        return 1
    out = pd.concat(frames, ignore_index=True)
    out_dir = paths.stat_dir() / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gb_vs_statistical.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
