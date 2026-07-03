# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Consolidate the baseline non-verbal ability -> vocabulary adjusted association (issue #186, Q4).

Surfaces ``gamma_blocks`` -- the block-design (non-verbal ability) coefficient the
ability-adjusted ITT models (LRPITT17-22) already fit -- across the six vocabulary
outcomes as a single Q4 read-out. ``gamma_blocks`` is the *partial* association of
baseline non-verbal ability with each vocabulary outcome, holding the child's own
baseline, linear age and randomised arm fixed.

Per the locked DAG (``notes/202606231600-dag-revision-consolidated.md``) block
design is an off-DAG, pre-randomisation child covariate and latent general ability
(``GA``) is the unobserved common cause, so this is an **adjusted association**
(block design is an ability proxy, confounded by ``GA``), **never a causal effect**.
Because the model already conditions on the child's own baseline vocabulary, the
coefficient is the *incremental* predictive value of non-verbal ability beyond
baseline vocabulary and age -- not the raw (marginal) correlation.

For each model it reports, on the logit scale (per +1 SD of block design): the
posterior median, 50/90/95% equal-tailed intervals, ``P(beta > 0)`` and the
round-odds evidence label; plus an items-scale average marginal effect (expected
vocabulary items gained per +1 SD of non-verbal ability, at the sample operating
point). Writes a consolidated CSV and a forest figure.

Usage::

    python scripts/ability_vocab_association.py --config dev
"""

from __future__ import annotations

import argparse
import importlib
import os

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from language_reading_predictors import paths
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.reporting import (
    evidence_label,
    favoured_direction,
)

# The ability-adjusted ITT family (LRPITT17-24), restricted to the six vocabulary
# outcomes (LRPITT23/24 are the L/W reading companions -- add them with --models
# for prognostic context).
DEFAULT_MODELS = (
    "lrpitt17",  # TR - taught receptive vocabulary (b1retau)
    "lrpitt18",  # TE - taught expressive vocabulary (b1extau)
    "lrpitt19",  # UR - not-taught receptive vocabulary (b1rent)
    "lrpitt20",  # UE - not-taught expressive vocabulary (b1exnt)
    "lrpitt21",  # R  - standardised receptive vocabulary (rowpvt)
    "lrpitt22",  # E  - standardised expressive vocabulary (eowpvt)
)
ADJUSTER = "gamma_blocks"


def _spec(model_id: str):
    mod = importlib.import_module(
        f"language_reading_predictors.statistical_models.{model_id}"
    )
    return mod.SPEC


def _eti(draws: np.ndarray, prob: float) -> tuple[float, float]:
    lo = (1.0 - prob) / 2.0
    return float(np.quantile(draws, lo)), float(np.quantile(draws, 1.0 - lo))


def summarise_gamma(
    g: np.ndarray, eta: np.ndarray | None, n_trials: int
) -> dict[str, object]:
    """Summarise a standardised-covariate coefficient's adjusted association.

    ``g`` is the posterior draws ``(S,)`` of the coefficient on ``z(blocks)`` (per
    +1 SD block design, logit scale). ``eta`` is the per-observation linear
    predictor draws ``(n_obs, S)`` (or ``None`` to skip the items-scale AME).
    Returns the logit summary (median, 50/90/95% equal-tailed intervals, ``P>0``,
    round-odds evidence + favoured-direction labels) and, when ``eta`` is given, the
    items-scale average marginal effect of a +1 SD shift: since ``eta`` already
    carries ``g * z(blocks)``, a +1 SD shift adds ``g`` to the linear predictor, so
    the per-draw AME is ``mean_i[expit(eta_i + g) - expit(eta_i)] * n_trials`` (the
    same AME logic the ITT/ROPE reporting uses for tau).
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    prob_pos = float((g > 0).mean())
    fav = favoured_direction(prob_pos)
    out: dict[str, object] = {
        "n_trials": int(n_trials),
        "gamma_logit_median": float(np.median(g)),
        "prob_positive": prob_pos,
        "evidence_label": evidence_label(prob_pos),
        "favoured_direction": fav["favoured_direction"],
        "favoured_label": fav["favoured_direction_label"],
    }
    out["gamma_logit_lo50"], out["gamma_logit_hi50"] = _eti(g, 0.50)
    out["gamma_logit_lo90"], out["gamma_logit_hi90"] = _eti(g, 0.90)
    out["gamma_logit_lo95"], out["gamma_logit_hi95"] = _eti(g, 0.95)
    if eta is not None:
        eta = np.asarray(eta, dtype=float)
        ame = (expit(eta + g[None, :]) - expit(eta)).mean(axis=0) * int(n_trials)  # (S,)
        out["items_ame_median"] = float(np.median(ame))
        out["items_ame_lo90"], out["items_ame_hi90"] = _eti(ame, 0.90)
    return out


def read_model(model_id: str, config: str) -> dict[str, object]:
    """Extract the ``gamma_blocks`` adjusted-association summary for one model."""
    spec = _spec(model_id)
    symbol = spec.outcome_symbol
    measure = MEASURES[symbol]
    model_dir = os.path.join(
        paths.output_root(), "statistical_models", "models", f"{model_id}-{config}"
    )
    trace = az.from_netcdf(os.path.join(model_dir, "trace.nc"))
    post = trace.posterior
    if ADJUSTER not in post:
        raise KeyError(
            f"{model_id}: {ADJUSTER!r} not in posterior -- is this an "
            "ability-adjusted (adjust_for=('blocks',)) model?"
        )
    g = post[ADJUSTER].stack(sample=("chain", "draw")).values  # (S,)
    eta = None
    if "eta" in post:
        eta = (
            post["eta"].stack(sample=("chain", "draw")).transpose("obs_id", "sample").values
        )  # (n_obs, S)
    return {
        "model_id": model_id,
        "outcome": symbol,
        "label": getattr(measure, "label", symbol),
        **summarise_gamma(g, eta, int(measure.n_trials)),
    }


def _forest(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1.5))
    med = df["gamma_logit_median"].to_numpy()
    ax.hlines(y, df["gamma_logit_lo95"], df["gamma_logit_hi95"], color="0.7", lw=1.5)
    ax.hlines(y, df["gamma_logit_lo50"], df["gamma_logit_hi50"], color="C0", lw=4)
    ax.plot(med, y, "o", color="C0", zorder=3)
    ax.axvline(0.0, ls="--", lw=1, color="grey")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.outcome} — {r.label}" for r in df.itertuples()])
    ax.set_xlabel("gamma_blocks (logit, per +1 SD block design) — adjusted association")
    ax.set_title("Baseline non-verbal ability → vocabulary (adjusted for own baseline, age, arm)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(os.path.splitext(path)[0] + ".svg")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Q4 read-out: baseline non-verbal ability -> vocabulary adjusted association (#186)."
    )
    ap.add_argument("--config", default="dev", help="Sampling config the models were fit at.")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument(
        "--out",
        default=None,
        help="CSV path (default: <output_root>/statistical_models/comparisons/ability_vocab_association.csv).",
    )
    args = ap.parse_args()

    df = pd.DataFrame([read_model(m, args.config) for m in args.models])
    out = args.out or os.path.join(
        paths.output_root(),
        "statistical_models",
        "comparisons",
        "ability_vocab_association.csv",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    _forest(df, os.path.splitext(out)[0] + ".png")

    cols = [
        "outcome", "label", "gamma_logit_median",
        "gamma_logit_lo90", "gamma_logit_hi90", "prob_positive", "evidence_label",
    ]
    if "items_ame_median" in df.columns:
        cols += ["items_ame_median"]
    print(df[cols].to_string(index=False))
    print(f"\n[written] {out}")
    print(f"[written] {os.path.splitext(out)[0]}.png / .svg")


if __name__ == "__main__":
    main()
