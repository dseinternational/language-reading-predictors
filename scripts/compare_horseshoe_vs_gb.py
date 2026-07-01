# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare a regularized-horseshoe predictor ranking (LRPHS) to the gradient-boosting
cluster ranking (#116 Phase E).

Both rankings are reduced to **construct level** and aligned:

- The horseshoe ranks measure *symbols* directly (``predictor_ranking.csv`` from
  ``scripts/fit_statistical_model.py lrphs0N``), keyed by posterior
  ``P(|beta| > delta)``.
- The gradient-boosting ranking is per raw column (``predictor_ranking.csv`` from
  ``scripts/rank_predictors.py --model lrpgb...``, ``member`` / ``perm_imp_mean``).
  Each column is mapped back to its construct symbol via :data:`MEASURES` (plus the
  age / blocks / behaviour covariate constructs); construct importance is the
  **max** per-column permutation importance among the construct's columns, matching
  the "best representative" logic of the GB cluster table.

The shared constructs are then compared by Spearman rank correlation and top-k
overlap. Broad agreement is reassurance that the ranking reflects signal in the
data rather than a tree-method artefact; divergences flag constructs whose apparent
importance is method-dependent.

Usage::

    python scripts/compare_horseshoe_vs_gb.py \\
        --horseshoe output/statistical_models/models/lrphs01-reporting/predictor_ranking.csv \\
        --gb output/ranking/lrpgbg12/predictor_ranking.csv \\
        --out output/statistical_models/models/lrphs01-reporting/horseshoe_vs_gb.csv

Writes ``horseshoe_vs_gb.csv`` (one row per shared construct) and prints the
Spearman correlation + top-k overlap for the dated ``notes/`` entry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Covariate constructs that are not MEASURES but do appear as horseshoe predictors
# and GB columns. Keyed by the raw column stem used on the GB side.
_COVARIATE_SYMBOLS = {"age": "age", "blocks": "blocks", "behav": "behav"}


def column_to_symbol_map() -> dict[str, str]:
    """Map each raw feature-column stem to its construct symbol.

    Built from :data:`MEASURES` (``measure.column -> symbol``) plus the covariate
    constructs. Imported lazily so the mapping helpers stay unit-testable without a
    heavy import when the caller passes an explicit map.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    mapping = {measure.column: symbol for symbol, measure in MEASURES.items()}
    mapping.update(_COVARIATE_SYMBOLS)
    return mapping


def member_to_symbol(member: str, col2sym: dict[str, str]) -> str | None:
    """Resolve a GB feature name to its construct symbol.

    Tries an exact column match first, then a substring match so wave/baseline
    suffixes or prefixes (e.g. ``yarclet_t1``, ``t1_yarclet``) still resolve. The
    longest matching stem wins, so ``b1retau`` is not shadowed by a shorter stem.
    Returns ``None`` for demographic-only / unmapped columns.
    """
    m = str(member).strip().lower()
    if m in col2sym:
        return col2sym[m]
    hits = [(stem, sym) for stem, sym in col2sym.items() if stem in m]
    if not hits:
        return None
    return max(hits, key=lambda kv: len(kv[0]))[1]


def gb_construct_ranking(gb: pd.DataFrame, col2sym: dict[str, str]) -> pd.DataFrame:
    """Collapse a per-column GB ranking to construct level.

    ``gb`` needs ``member`` + ``perm_imp_mean``. Construct importance is the max
    per-column permutation importance among the construct's columns (the GB "best
    representative"); constructs are then ranked by that value (1 = most important).
    """
    df = gb[["member", "perm_imp_mean"]].copy()
    df["symbol"] = df["member"].map(lambda x: member_to_symbol(x, col2sym))
    df = df.dropna(subset=["symbol"])
    grouped = (
        df.groupby("symbol", as_index=False)["perm_imp_mean"]
        .max()
        .rename(columns={"perm_imp_mean": "gb_perm_imp"})
    )
    grouped["gb_rank"] = (
        grouped["gb_perm_imp"].rank(ascending=False, method="min").astype(int)
    )
    return grouped.sort_values("gb_rank").reset_index(drop=True)


def compare_rankings(
    horseshoe: pd.DataFrame,
    gb: pd.DataFrame,
    *,
    col2sym: dict[str, str] | None = None,
    topk: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Align a horseshoe construct ranking to a GB construct ranking.

    ``horseshoe`` needs ``predictor`` + ``p_abs_gt_delta`` (+ optional ``rank``);
    ``gb`` needs ``member`` + ``perm_imp_mean``. Returns the merged per-construct
    table (shared constructs only) and a summary dict (Spearman rho/p, top-k
    overlap, counts).
    """
    if col2sym is None:
        col2sym = column_to_symbol_map()

    hs = horseshoe.copy()
    if "rank" not in hs.columns:
        hs["rank"] = (
            hs["p_abs_gt_delta"].rank(ascending=False, method="min").astype(int)
        )
    hs = hs.rename(columns={"rank": "hs_rank"})[
        ["predictor", "hs_rank", "p_abs_gt_delta"]
    ]

    gb_c = gb_construct_ranking(gb, col2sym)
    merged = hs.merge(
        gb_c, left_on="predictor", right_on="symbol", how="inner"
    ).drop(columns="symbol")
    merged = merged.sort_values("hs_rank").reset_index(drop=True)

    n = len(merged)
    if n >= 3:
        # Spearman rho via the Pearson correlation of the two rank vectors (no SciPy
        # dependency): identical for distinct integer ranks.
        rho = float(np.corrcoef(merged["hs_rank"], merged["gb_rank"])[0, 1])
    else:
        rho = float("nan")

    hs_top = set(hs.sort_values("hs_rank")["predictor"].head(topk))
    gb_top = set(gb_c.sort_values("gb_rank")["symbol"].head(topk))
    overlap = sorted(hs_top & gb_top)

    summary = {
        "shared_constructs": n,
        "spearman_rho": rho,
        "topk": topk,
        "topk_overlap": len(overlap),
        "topk_overlap_symbols": overlap,
        "hs_topk": sorted(hs_top),
        "gb_topk": sorted(gb_top),
    }
    return merged, summary


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: file not found: {p}")
    return pd.read_csv(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horseshoe", required=True, help="LRPHS predictor_ranking.csv")
    ap.add_argument("--gb", required=True, help="GB predictor_ranking.csv")
    ap.add_argument("--out", required=True, help="output horseshoe_vs_gb.csv")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args(argv)

    hs = _read_csv(args.horseshoe)
    gb = _read_csv(args.gb)
    merged, summary = compare_rankings(hs, gb, topk=args.topk)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    rho = summary["spearman_rho"]
    print(f"shared constructs: {summary['shared_constructs']}")
    print(
        "Spearman rho(hs_rank, gb_rank) = "
        + ("n/a (<3 shared)" if np.isnan(rho) else f"{rho:+.3f}")
    )
    print(
        f"top-{summary['topk']} overlap: "
        f"{summary['topk_overlap']}/{summary['topk']} "
        f"{summary['topk_overlap_symbols']}"
    )
    print(f"  horseshoe top-{summary['topk']}: {summary['hs_topk']}")
    print(f"  gb        top-{summary['topk']}: {summary['gb_topk']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
