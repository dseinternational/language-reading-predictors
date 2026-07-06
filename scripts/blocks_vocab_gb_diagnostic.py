# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""GB corroboration for issue #186 (Q4, Phase 2): does block design carry nonparametric predictive signal for vocabulary?

For each vocabulary LEVEL gradient-boosting model (LRPGBL01–06) this refits an
ad-hoc variant with block design added to the predictor set — `blocks` is normally
in `DEFAULT_EXCLUDED` (t1-only), and `data_utils.load_data` now broadcasts it per
child so it is a usable time-invariant covariate — and reports where `blocks` ranks
by out-of-fold permutation importance.

This is a nonparametric cross-check of the Bayesian `gamma_blocks` read-out
(`scripts/ability_vocab_association.py`). The interpretations differ on purpose:
unlike the ANCOVA-on-logit ITT models, the GB model does not privilege the child's
own baseline vocabulary as an anchor, so a mid-pack `blocks` here reflects the
**marginal** predictive contribution of non-verbal ability (larger, and shared with
baseline vocabulary), not the **incremental** value beyond baseline that the
Bayesian coefficient isolates. Agreement on *where* `blocks` matters (taught vs
standardised) is the corroboration; the magnitude gap is expected.

The variant configs are built with ``dataclasses.replace`` and are **not**
registered in ``MODELS`` (no new permanent models). Direction is the marginal
Spearman of block design with the target (positive as expected — the point of
interest is rank, not sign).

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
from language_reading_predictors import paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import MODELS
from language_reading_predictors.models.common import RunConfig

# Vocabulary LEVEL GB models: taught/not-taught receptive & expressive + standardised.
VOCAB_LEVEL_MODELS = (
    "lrpgbl01",  # b1retau - taught receptive
    "lrpgbl02",  # b1extau - taught expressive
    "lrpgbl03",  # b1rent  - not-taught receptive
    "lrpgbl04",  # b1exnt  - not-taught expressive
    "lrpgbl05",  # rowpvt  - standardised receptive
    "lrpgbl06",  # eowpvt  - standardised expressive
)


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
