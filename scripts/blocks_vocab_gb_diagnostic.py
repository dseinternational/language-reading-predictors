# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rank block design in the vocabulary level gradient-boosting models.

Refits each model with block design in its predictor set and reports its rank by
out-of-fold permutation importance. The loader broadcasts this time-invariant
measure to each child's waves. Any added variant remains outside the registry.

Permutation importance describes prediction in a model with all its other
predictors. It is not a marginal association or a causal effect. The script also
reports the unadjusted Spearman correlation with the target. Neither quantity is
the baseline-adjusted ``gamma_blocks`` coefficient reported by
``scripts/ability_vocab_association.py``; their magnitudes are not comparable.

Usage::

    python scripts/blocks_vocab_gb_diagnostic.py --config dev
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import pandas as pd
from scipy.stats import spearmanr

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors import figure_io, model_ids, paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import MODELS
from language_reading_predictors.models.common import RunConfig

# Vocabulary LEVEL GB models: taught/not-taught receptive & expressive + standardised.
# _resolve accepts canonical and legacy model IDs.
VOCAB_LEVEL_MODELS = (
    "lrp-rli-gbl-001",  # b1retau - taught receptive
    "lrp-rli-gbl-002",  # b1extau - taught expressive
    "lrp-rli-gbl-003",  # b1rent  - not-taught receptive
    "lrp-rli-gbl-004",  # b1exnt  - not-taught expressive
    "lrp-rli-gbl-005",  # rowpvt  - standardised receptive
    "lrp-rli-gbl-006",  # eowpvt  - standardised expressive
)


def _resolve(model_id: str) -> str:
    """Resolve a user-supplied id (legacy or canonical, any form) to its registry key.

    Returns the input unchanged when unrecognised so ``MODELS[...]`` raises the
    usual ``KeyError`` for an unknown model.
    """
    aliases: dict[str, str] = {}
    for key in MODELS:
        aliases[key.lower()] = key
        try:
            mid = model_ids.parse_canonical(key)
        except model_ids.ModelIdError:
            continue
        for form in (mid.legacy, mid.display, mid.module):
            aliases[form.lower()] = key
    return aliases.get(model_id.strip().lower(), model_id)


def blocks_rank(perm: pd.DataFrame) -> tuple[int | None, float | None, int]:
    """Rank + importance of block design in a permutation-importance table.

    ``perm`` is ``permutation_importance.csv`` (columns ``feature`` /
    ``importance_mean``). Returns ``(rank, importance_mean, n_predictors)`` with a
    1-based rank by descending mean importance, or ``(None, None, n)`` if block
    design is absent.
    """
    fcol = "feature" if "feature" in perm.columns else perm.columns[0]
    ordered = perm.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    hit = ordered.index[ordered[fcol] == V.BLOCKS].tolist()
    if not hit:
        return None, None, len(ordered)
    i = hit[0]
    return i + 1, float(ordered.loc[i, "importance_mean"]), len(ordered)


def run_one(base_id: str, config: str, df: pd.DataFrame) -> dict[str, object]:
    import language_reading_predictors.models  # noqa: F401  (populates MODELS)

    base_id = _resolve(base_id)
    cfg = MODELS[base_id]
    target = cfg.target_var
    if V.BLOCKS not in cfg.predictor_vars:
        cfg = dataclasses.replace(
            cfg,
            predictor_vars=[V.BLOCKS, *cfg.predictor_vars],
            model_id=f"{base_id}_blocks",
        )
    ctx = cfg.pipeline_cls(cfg, RunConfig.from_name(config)).fit()
    perm = pd.read_csv(os.path.join(ctx.output_dir, "permutation_importance.csv"))
    rank, imp, n = blocks_rank(perm)
    sub = df[[V.BLOCKS, target]].dropna()
    rho = float(spearmanr(sub[V.BLOCKS], sub[target]).statistic) if len(sub) > 2 else None
    return {
        "model_id": base_id,
        "target": target,
        "n_predictors": n,
        "blocks_perm_rank": rank,
        "blocks_perm_importance": imp,
        "blocks_target_spearman": rho,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GB corroboration for #186 Q4: block design's permutation-importance rank in the vocab level models."
    )
    ap.add_argument("--config", default="dev", help="GB run config (dev is enough for a rank).")
    ap.add_argument("--models", nargs="+", default=list(VOCAB_LEVEL_MODELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    figure_io.use_house_style()  # the fits draw figures, as under scripts/fit_model.py

    df = data_utils.load_data()
    rows = [run_one(m, args.config, df) for m in args.models]
    out = args.out or os.path.join(
        paths.output_root(), "comparisons", "blocks_vocab_gb_diagnostic.csv"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
