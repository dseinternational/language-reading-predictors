# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproduce the evidence design analysis and ROPE-reporting figures.

Regenerates the two figures in
``notes/202606261304-evidence-strength-and-rope-reporting.md``:

1. **Type-S / Type-M design analysis** of the ITT suite (Gelman & Carlin 2014):
   for each graded single-outcome ITT model, the posterior mean ``tau`` and
   posterior SD ``s`` drive the sign-error (Type-S) and magnitude-exaggeration
   (Type-M) curves at the study's n.
2. **ROPE-anchored continuous reporting** for letter sounds (L) vs word reading
   (W): the items-scale posteriors with the region of practical equivalence, and
   ``P(effect > delta)`` as the minimally-important difference rises.

Models are refit (``--draws`` posterior draws; a quick dev-style sample by
default) rather than read from disk, so the script is self-contained. It uses
:func:`reporting.rope_summary` and the adopted minimally-important differences
:data:`measures.ROPE_DELTA`. Figures are written to
``output/statistical_models/models/design_analysis/`` (git-ignored); ``--refresh-note``
also copies them into ``notes/assets/`` to update the committed note.

Usage::

    python scripts/design_analysis.py
    python scripts/design_analysis.py --draws 1500 --refresh-note
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.factories import build_itt_model
from language_reading_predictors.statistical_models.measures import MEASURES, ROPE_DELTA
from language_reading_predictors.statistical_models.preprocessing import load_and_prepare

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# Graded single-outcome ITT models, by module / symbol. Floored P, N (off-floor
# Bernoulli tau) are excluded — they live on a probability scale.
GRADED = [
    ("lrp-rli-itt-007", "L"),
    ("lrp-rli-itt-010", "W"),
    ("lrp-rli-itt-008", "B"),
    ("lrp-rli-itt-002", "TE"),
    ("lrp-rli-itt-001", "TR"),
    ("lrp-rli-itt-004", "UE"),
    ("lrp-rli-itt-003", "UR"),
    ("lrp-rli-itt-006", "E"),
    ("lrp-rli-itt-005", "R"),
]
LABELS = {m.symbol: m.label for m in MEASURES.values()}

C_STRONG, C_MOD, C_WEAK, C_CURVE = "#1b7837", "#f1a340", "#999999", "#2166ac"
LAM_STRONG = stats.norm.ppf(0.975)  # 1.96 -> pd 0.975
LAM_MOD = stats.norm.ppf(0.90)  # 1.28 -> pd 0.90


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def _prepare_and_build(spec):
    extra = spec.extra
    kw = dict(
        phase_mode="itt",
        covariates=tuple(extra.get("adjust_for", ())),
        restrict_complete=tuple(extra.get("restrict_complete", ())),
        drop_missing_pre=bool(extra.get("drop_missing_pre", True)),
        pre_required=(
            tuple(extra["pre_required"]) if extra.get("pre_required") is not None else None
        ),
    )
    if extra.get("outcomes") is not None:
        kw["outcomes"] = tuple(extra["outcomes"])
    prepared = load_and_prepare(**kw)
    return build_itt_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        use_age_gp=extra.get("use_age_gp", False),
        use_own_baseline_gp=extra.get("use_own_baseline_gp", False),
        adjust_for=tuple(extra.get("adjust_for", ())),
        cross_symbols=extra.get("cross_symbols"),
        use_age_linear=extra.get("use_age_linear", False),
        use_own_baseline=extra.get("use_own_baseline", True),
    )


def fit_outcome(mod_name, sym, draws, tune, chains, seed):
    module = __import__(
        f"language_reading_predictors.statistical_models.{mod_name}", fromlist=["SPEC"]
    )
    built = _prepare_and_build(module.SPEC)
    with built.model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            progressbar=False,
            random_seed=seed,
            compute_convergence_checks=False,
        )
    # SimpleNamespace-like trace wrapper for the reporting helpers (.posterior).
    trace = type("T", (), {"posterior": idata.posterior})()
    tau = np.asarray(idata.posterior["tau"]).ravel()
    G = np.asarray(built.prepared.G)
    n_trials = int(built.prepared.n_trials[sym])
    # Per-draw items-scale average marginal effect (shared core).
    _, ame_prob = _report._itt_ame_draws(trace, G=G)
    items = ame_prob * n_trials
    return dict(
        sym=sym,
        trace=trace,
        G=G,
        n_trials=n_trials,
        tau=tau,
        items=items,
        mean=float(tau.mean()),
        sd=float(tau.std(ddof=1)),
        n=int(G.shape[0]),
    )


# --------------------------------------------------------------------------- #
# Design analysis (Gelman & Carlin 2014)
# --------------------------------------------------------------------------- #
def retrodesign(lam, alpha=0.05):
    """Power, Type-S, Type-M for a true-effect/SE ratio ``lam`` (s := 1)."""
    a = abs(float(lam))
    z = stats.norm.ppf(1 - alpha / 2.0)
    p_hi = 1.0 - stats.norm.cdf(z - a)
    p_lo = stats.norm.cdf(-z - a)
    power = p_hi + p_lo
    type_s = p_lo / power if power > 0 else np.nan
    e_hi = a * p_hi + stats.norm.pdf(z - a)
    e_lo = -a * p_lo + stats.norm.pdf(z + a)
    exagg = (e_hi + e_lo) / (power * a) if (power > 0 and a > 0) else np.nan
    return power, type_s, exagg


def _tier_color(lam):
    if lam >= LAM_STRONG:
        return C_STRONG
    if lam >= LAM_MOD:
        return C_MOD
    return C_WEAK


def figure_design_analysis(results, out_png, out_pdf):
    label_off = {
        "L": (7, 5), "W": (8, 9), "TE": (9, -14), "B": (-20, 10),
        "TR": (8, 7), "UR": (-22, -14), "UE": (8, 5),
    }
    xs = np.linspace(0.4, 4.0, 500)
    typem = np.array([retrodesign(x)[2] for x in xs])
    types = np.array([retrodesign(x)[1] for x in xs]) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.6, 8.8), sharex=True, gridspec_kw={"hspace": 0.13}
    )

    def bands(ax):
        ax.axvspan(LAM_STRONG, 4.0, color=C_STRONG, alpha=0.06)
        ax.axvspan(LAM_MOD, LAM_STRONG, color=C_MOD, alpha=0.07)
        ax.axvspan(0.4, LAM_MOD, color=C_WEAK, alpha=0.07)
        ax.axvline(LAM_STRONG, color=C_STRONG, lw=1.0, ls="--", alpha=0.8)
        ax.axvline(LAM_MOD, color=C_MOD, lw=1.0, ls="--", alpha=0.8)

    bands(ax1)
    ax1.axhline(1.0, color="#444", lw=0.9, ls=":")
    ax1.plot(xs, typem, color=C_CURVE, lw=2.4)
    ax1.set_ylim(0.9, 3.4)
    ax1.set_ylabel("Type-M (exaggeration ratio)")
    for r in results:
        lam = abs(r["mean"] / r["sd"])
        if lam > 3.9:
            continue
        _, _, m = retrodesign(lam)
        col = _tier_color(lam)
        ax1.scatter([lam], [m], s=70, color=col, edgecolor="white", lw=1.1, zorder=5)
        ax1.annotate(
            r["sym"], (lam, m), textcoords="offset points",
            xytext=label_off.get(r["sym"], (6, 5)), fontsize=9.5,
            fontweight="bold", color=col,
        )

    bands(ax2)
    ax2.plot(xs, types, color=C_CURVE, lw=2.4)
    ax2.set_ylim(-0.4, 9.5)
    ax2.set_ylabel("Type-S (sign-error rate, %)")
    ax2.set_xlabel("true effect in SE units,  |tau| / s")
    for r in results:
        lam = abs(r["mean"] / r["sd"])
        if lam > 3.9:
            continue
        _, ts, _ = retrodesign(lam)
        col = _tier_color(lam)
        ax2.scatter([lam], [ts * 100], s=70, color=col, edgecolor="white", lw=1.1, zorder=5)

    for ax in (ax1, ax2):
        ax.set_xlim(0.4, 4.0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Type-S / Type-M design analysis of the ITT suite", y=0.97)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def figure_rope(results, out_png, out_pdf):
    by = {r["sym"]: r for r in results}
    colors = {"L": "#1b7837", "W": "#762a83"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    pair = [by["L"], by["W"]]
    xmax = max(np.quantile(r["items"], 0.995) for r in pair) + 0.5

    xgrid = np.linspace(-1.0, xmax, 400)
    # Each outcome has its own adopted ROPE half-width (items scale): the shared
    # ±1 band was wrong for L, whose adopted delta is 2 items. Draw a per-outcome
    # band (matched to the outcome's colour) so the plot reflects ROPE_DELTA.
    axL.axvline(0, color="#444", lw=1.0, ls=":")
    for r in pair:
        sym = r["sym"]
        delta = ROPE_DELTA[sym]
        item_word = "item" if delta == 1 else "items"
        axL.axvspan(-delta, delta, color=colors[sym], alpha=0.10,
                    label=f"{sym} ROPE (|effect| < {delta:g} {item_word})")
        kde = stats.gaussian_kde(r["items"])
        axL.plot(xgrid, kde(xgrid), color=colors[sym], lw=2.4,
                 label=f"{sym} {LABELS[sym]}")
        axL.fill_between(xgrid, kde(xgrid), color=colors[sym], alpha=0.12)
    axL.set_xlabel("treatment effect (extra test items correct)")
    axL.set_ylabel("posterior density")
    axL.set_title("Posterior of the effect (items), with ROPE")
    axL.legend(fontsize=9, frameon=False, loc="upper right")

    dgrid = np.linspace(0.0, xmax, 300)
    for r in pair:
        p_exceed = np.array([(r["items"] > d).mean() for d in dgrid])
        axR.plot(dgrid, p_exceed, color=colors[r["sym"]], lw=2.4, label=r["sym"])
    axR.axhline(0.975, color="#888", lw=1.0, ls="--", label="pd = 0.975")
    axR.set_ylim(0, 1.02)
    axR.set_xlabel("minimally-important difference, delta (items)")
    axR.set_ylabel("P(treatment effect > delta)")
    axR.set_title("Probability of a benefit of at least delta items")
    axR.legend(fontsize=9.5, frameon=False, loc="upper right")
    for ax in (axL, axR):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("ROPE-anchored continuous reporting: letter sounds (L) vs word reading (W)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def print_cards(results):
    by = {r["sym"]: r for r in results}
    print("\n" + "=" * 78)
    print("ROPE summaries (reporting.rope_summary, adopted delta from ROPE_DELTA)")
    print("=" * 78)
    for sym in ("L", "W"):
        r = by[sym]
        delta = ROPE_DELTA[sym]
        s = _report.rope_summary(r["trace"], G=r["G"], n_trials=r["n_trials"], delta=delta)
        print(
            f"{sym} {LABELS[sym]} (n={r['n']}, {r['n_trials']} items, delta={delta:g})\n"
            f"  items: median {s['items_median']:+.2f}  "
            f"95% [{s['items_lo']:+.2f}, {s['items_hi']:+.2f}]\n"
            f"  pd = {s['pd']:.3f} ({s['direction_label']});  "
            f"P(benefit>=delta) = {s['prob_benefit_ge_delta']:.3f} ({s['benefit_label']});  "
            f"in-ROPE = {s['prob_in_rope']:.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument(
        "--refresh-note",
        action="store_true",
        help="also copy the figures into notes/assets/ to update the committed note",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR). Default: repo-local output/."
        ),
    )
    args = ap.parse_args()
    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")

    out_dir = os.path.join(str(_paths.stat_models_dir()), "design_analysis")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for mod_name, sym in GRADED:
        print(f"fitting {sym} ({mod_name}) ...")
        results.append(fit_outcome(mod_name, sym, args.draws, args.tune, args.chains, args.seed))

    print("\n" + "=" * 78)
    print(f"{'outcome':26s} {'n':>4s} {'tau':>8s} {'s':>7s} {'|t|/s':>7s}")
    for r in sorted(results, key=lambda x: -abs(x["mean"] / x["sd"])):
        print(f"{r['sym']+' '+LABELS[r['sym']]:26.26s} {r['n']:4d} "
              f"{r['mean']:8.3f} {r['sd']:7.3f} {abs(r['mean']/r['sd']):7.2f}")

    da_png = os.path.join(out_dir, "type_s_m_design_analysis.png")
    da_pdf = os.path.join(out_dir, "type_s_m_design_analysis.pdf")
    rope_png = os.path.join(out_dir, "rope_continuous_reporting.png")
    rope_pdf = os.path.join(out_dir, "rope_continuous_reporting.pdf")
    figure_design_analysis(results, da_png, da_pdf)
    figure_rope(results, rope_png, rope_pdf)
    print_cards(results)
    print(f"\nWROTE {da_png}\nWROTE {rope_png}")

    if args.refresh_note:
        notes_assets = os.path.join(str(_paths.DOCS_DIR), "..", "notes", "assets")
        notes_assets = os.path.normpath(notes_assets)
        os.makedirs(notes_assets, exist_ok=True)
        shutil.copy(da_png, os.path.join(notes_assets, "202606261304-type-s-m-design-analysis.png"))
        shutil.copy(rope_png, os.path.join(notes_assets, "202606261304-rope-continuous-reporting.png"))
        print(f"refreshed note figures in {notes_assets}")


if __name__ == "__main__":
    main()
