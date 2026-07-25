# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Is there a MINIMUM word-reading level below which children do not progress?

Asks, for three code-route targets - letter sounds (``L``, yarclet, 32 items),
phoneme blending (``B``, blending, 10 items) and nonword reading (``N``, nonword,
6 items) - whether progress over the following period depends on the child's
word-reading level at the START of that period, and whether that dependence is a
THRESHOLD (nothing below some level, then progress) or a GRADIENT (more reading,
more progress, no special point).

Three passes:

P1  ROOM-RESTRICTED band tables. Every row is one child-period transition
    (t -> t+1, three per child). Rows are kept only where the target has real
    headroom, because "did not progress" and "had nowhere to go" are the same
    number otherwise: letter sounds are at a median 28 of 32 by t3 in the top
    word-reading half. Progress is then contrasted across bands of the
    period-start word-reading score.

P2  THE EXISTENCE CHECK. The literal form of the question: among children who
    read NO words at the start of a period, does anyone progress? A single
    counter-example refutes a strict prerequisite, so this pass reports raw
    counts rather than effect sizes. For nonword reading it also reports the
    floor-exit rate (0 -> >=1) by band, since for a 6-item test with a median of
    1 at t4 "progress" is mostly floor exit.

P3  THRESHOLD VERSUS GRADIENT, model-based. Per target, four Beta-Binomial
    ANCOVA specifications of the period-end count given the period-start count,
    differing ONLY in how period-start word reading enters:
      null    - not at all
      linear  - straight line in words
      log     - straight line in log1p(words) (diminishing returns)
      hinge   - flat below a FREE breakpoint theta, then a straight line
    Compared by PSIS-LOO. The deliverable is (a) whether the hinge wins, and
    (b) the posterior for theta in words - a threshold nobody can locate is not
    a threshold, so an unresolved theta is a real answer and is reported as one.

IDENTIFICATION, stated once and applying throughout. Only the ITT arm contrast is
randomised in this study; nothing here is causal. Under the lagged DAG
(``dag/dag-language-reading-lagged.dagitty``) the coupling ``WR -> LS`` is
identifiable with a small set - age, hearing, own letter-sound baseline, speech -
and that set is used below. ``WR -> PA`` needs 8-11 adjusters (unfittable at this
sample size) and ``WR -> NW`` is NOT AN EDGE IN THE GRAPH AT ALL (see #428). The
same covariate set is therefore applied to all three targets for comparability,
but only the letter-sound row is an adjusted association in the DAG's terms; the
blending and nonword rows are descriptive, under-adjusted, and must be read as
such. Latent general ability confounds all three and is unblockable.

MEASUREMENT, the standing caveat on every number here. Letter sounds have a
32-item ceiling that the strong readers reach; blending is 10 items administered
as three-alternative picture-pointing, so the chance floor is near 3.3 and 19% of
children are at ceiling by t4; nonword reading is 6 items, floored at zero for
72 / 64 / 52 / 40% of children at t1-t4. A "threshold" in any of these can be
manufactured by the instrument rather than the child.

Run with the conda env interpreter from the repo root; writes CSVs to ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from language_reading_predictors.data_utils import load_data

CI = 0.89  # house standard (notes/202607172359-credible-interval-standard.md)
N_BOOT = 4000
SEED = 20260724
WR = "ewrswr"  # word reading, 79 items (with the >25-word instrument switch)

# target -> (column, n_items, headroom cut, "meaningful gain" in items)
#
# The headroom cut keeps rows where the child could gain at least a quarter of
# the scale. Sensitivity to the cut is reported in P1 (a looser cut is run
# alongside), because the cut is a judgement call and it moves the sample.
TARGETS: dict[str, dict] = {
    "L": {"col": "yarclet", "n_items": 32, "room_at_or_below": 24, "meaningful": 2,
          "label": "letter sounds"},
    "B": {"col": "blending", "n_items": 10, "room_at_or_below": 7, "meaningful": 1,
          "label": "phoneme blending"},
    "N": {"col": "nonword", "n_items": 6, "room_at_or_below": 4, "meaningful": 1,
          "label": "nonword reading"},
}

# Period-start word-reading bands. The 25-word boundary is not arbitrary: children
# reading more than 25 words are given additional Test of Single-Word Reading
# items, so the upper tail of EWRSWR is partly a different instrument.
BANDS: list[tuple[str, float, float]] = [
    ("0 words", -0.5, 0.5),
    ("1-4", 0.5, 4.5),
    ("5-9", 4.5, 9.5),
    ("10-24", 9.5, 24.5),
    ("25+", 24.5, np.inf),
]

COVARIATES = ["age", "hearing_c", "deapp_c"]  # + own baseline, arm, on-intervention


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def transitions() -> pd.DataFrame:
    """One row per child-period transition (t -> t+1), t in 1..3."""
    raw = load_data()
    keep = raw[raw["time"] <= 3].copy()
    keep["wr_pre"] = keep[WR]
    keep["arm"] = (keep["group"] == 1).astype(int)  # 1 = immediate, 0 = wait-list
    keep["on_int"] = keep["on_intervention"].astype(int)
    for sym, spec in TARGETS.items():
        col = spec["col"]
        keep[f"{sym}_pre"] = keep[col]
        keep[f"{sym}_post"] = keep[f"{col}_next"]
        keep[f"{sym}_gain"] = keep[f"{col}_next"] - keep[col]
    return keep


def _band(x: float) -> str:
    for name, lo, hi in BANDS:
        if lo < x <= hi:
            return name
    return "NA"


def analysis_frame(df: pd.DataFrame, sym: str, room_cut: float | None) -> pd.DataFrame:
    """Rows with an observed transition on ``sym``, optionally room-restricted."""
    # Every target's period-start level is carried through (not just the focal
    # one) so P4 can add letter sounds as an adjuster without re-deriving frames.
    cols = ["subject_id", "time", "wr_pre", "arm", "on_int", *COVARIATES,
            *(f"{s}_pre" for s in TARGETS),
            f"{sym}_post", f"{sym}_gain"]
    out = df[cols].dropna(subset=["wr_pre", f"{sym}_pre", f"{sym}_post"]).copy()
    if room_cut is not None:
        out = out[out[f"{sym}_pre"] <= room_cut]
    out["band"] = out["wr_pre"].apply(_band)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# effect size (matches 202607241600-word-reading-band-comparisons.py)
# ---------------------------------------------------------------------------


def cliffs_delta(top: np.ndarray, bottom: np.ndarray) -> float:
    if len(top) == 0 or len(bottom) == 0:
        return np.nan
    return float(np.sign(top[:, None] - bottom[None, :]).mean())


def cliffs_delta_ci(top, bottom, rng, top_ids=None, bottom_ids=None):
    """Cliff's delta with a percentile 89% interval.

    The resample is CLUSTERED BY CHILD when ``top_ids`` / ``bottom_ids`` are given:
    each row here is one child-period transition and a child contributes up to
    three of them, so resampling rows independently would treat 143 correlated
    observations as 143 independent ones and report intervals that are too narrow.
    Children are drawn with replacement and all of a drawn child's rows come with
    them.
    """
    point = cliffs_delta(top, bottom)
    if not np.isfinite(point):
        return point, np.nan, np.nan

    def _resample(values, ids):
        if ids is None:
            return rng.choice(values, len(values), replace=True)
        uniq = np.unique(ids)
        drawn = rng.choice(uniq, len(uniq), replace=True)
        return np.concatenate([values[ids == c] for c in drawn])

    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = cliffs_delta(_resample(top, top_ids), _resample(bottom, bottom_ids))
    lo, hi = (1 - CI) / 2, 1 - (1 - CI) / 2
    return point, float(np.nanquantile(boots, lo)), float(np.nanquantile(boots, hi))


# ---------------------------------------------------------------------------
# P1 - room-restricted band tables
# ---------------------------------------------------------------------------


def run_p1(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for sym, spec in TARGETS.items():
        for cut_label, cut in (("room", spec["room_at_or_below"]),
                               ("loose", spec["n_items"] - 1),
                               ("all", None)):
            frame = analysis_frame(df, sym, cut)
            if frame.empty:
                continue
            # Residualise the gain on the period-start level of the TARGET, pooled
            # within this restriction: a raw gain contrast across word-reading bands
            # is regression to the mean plus a ceiling constraint, because the bands
            # differ on the target's own baseline.
            x = frame[f"{sym}_pre"].to_numpy(float)
            y = frame[f"{sym}_gain"].to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
            frame = frame.assign(resid=y - (intercept + slope * x))
            zero_band = frame[frame["band"] == "0 words"]
            base = zero_band["resid"].to_numpy()
            base_ids = zero_band["subject_id"].to_numpy()
            for name, _lo, _hi in BANDS:
                b = frame[frame["band"] == name]
                if b.empty:
                    continue
                d, dlo, dhi = ((np.nan, np.nan, np.nan) if name == "0 words"
                               else cliffs_delta_ci(b["resid"].to_numpy(), base, rng,
                                                    b["subject_id"].to_numpy(),
                                                    base_ids))
                rows.append({
                    "target": sym, "label": spec["label"], "restriction": cut_label,
                    "band": name, "n_rows": len(b), "n_children": b["subject_id"].nunique(),
                    "median_pre": float(b[f"{sym}_pre"].median()),
                    "median_gain": float(b[f"{sym}_gain"].median()),
                    "mean_gain": float(b[f"{sym}_gain"].mean()),
                    "pct_any_gain": float((b[f"{sym}_gain"] > 0).mean() * 100),
                    "pct_meaningful_gain":
                        float((b[f"{sym}_gain"] >= spec["meaningful"]).mean() * 100),
                    "mean_resid_gain": float(b["resid"].mean()),
                    "cliffs_delta_vs_zero_band": d,
                    "delta_lo": dlo, "delta_hi": dhi,
                })
    table = pd.DataFrame(rows)
    table.to_csv(out / "p1_band_progress.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# P2 - the existence check
# ---------------------------------------------------------------------------


def run_p2(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sym, spec in TARGETS.items():
        frame = analysis_frame(df, sym, spec["room_at_or_below"])
        zero = frame[frame["band"] == "0 words"]
        rows.append({
            "target": sym, "label": spec["label"],
            "n_rows_zero_word_readers": len(zero),
            "n_children": zero["subject_id"].nunique(),
            "n_any_gain": int((zero[f"{sym}_gain"] > 0).sum()),
            "n_meaningful_gain": int((zero[f"{sym}_gain"] >= spec["meaningful"]).sum()),
            "max_gain": float(zero[f"{sym}_gain"].max()) if len(zero) else np.nan,
            "median_gain": float(zero[f"{sym}_gain"].median()) if len(zero) else np.nan,
        })
    existence = pd.DataFrame(rows)
    existence.to_csv(out / "p2_zero_word_reader_progress.csv", index=False)

    # Floor exit, band by band, for the two floored targets.
    exits = []
    for sym in ("N", "B", "L"):
        spec = TARGETS[sym]
        frame = analysis_frame(df, sym, None)
        floor = 0 if sym != "B" else 3  # blending's chance floor is ~3.3 of 10
        at_floor = frame[frame[f"{sym}_pre"] <= floor]
        for name, _lo, _hi in BANDS:
            b = at_floor[at_floor["band"] == name]
            if b.empty:
                continue
            exits.append({
                "target": sym, "label": spec["label"], "floor_defined_as_le": floor,
                "band": name, "n_at_floor": len(b),
                "n_children": b["subject_id"].nunique(),
                "n_exited": int((b[f"{sym}_post"] > floor).sum()),
                "pct_exited": float((b[f"{sym}_post"] > floor).mean() * 100),
            })
    exit_table = pd.DataFrame(exits)
    exit_table.to_csv(out / "p2_floor_exit_by_band.csv", index=False)
    return existence, exit_table


# ---------------------------------------------------------------------------
# P3 - threshold versus gradient
# ---------------------------------------------------------------------------


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    sd = np.nanstd(a)
    return (a - np.nanmean(a)) / (sd if sd > 0 else 1.0)


def _logit_z(values: np.ndarray, n_items: int) -> np.ndarray:
    p = (np.asarray(values, dtype=float) + 0.5) / (n_items + 1.0)
    return _z(np.log(p / (1 - p)))


def _covariate_matrix(frame: pd.DataFrame, sym: str, n_items: int,
                      extra_symbols: tuple[str, ...] = ()):
    """Own-baseline logit + age + arm + on-intervention + hearing + speech.

    Hearing and speech are mean-filled with a missingness indicator, matching the
    suite's ``adjust_for`` convention rather than dropping a fifth of the rows.
    ``extra_symbols`` adds another measure's period-start logit - used by P4 to ask
    whether an apparent word-reading threshold is really a letter-sound threshold.
    """
    pre = frame[f"{sym}_pre"].to_numpy(float)
    p = (pre + 0.5) / (n_items + 1.0)
    cols = {"own_baseline_logit": _z(np.log(p / (1 - p))),
            "age": _z(frame["age"].to_numpy(float)),
            "arm": frame["arm"].to_numpy(float),
            "on_intervention": frame["on_int"].to_numpy(float)}
    for extra in extra_symbols:
        cols[f"{extra}_pre_logit"] = _logit_z(
            frame[f"{extra}_pre"].to_numpy(float), TARGETS[extra]["n_items"]
        )
    for name in ("hearing_c", "deapp_c"):
        v = frame[name].to_numpy(float)
        miss = np.isnan(v)
        filled = np.where(miss, np.nanmean(v), v)
        cols[name] = _z(filled)
        if miss.any():
            cols[f"{name}_missing"] = miss.astype(float)
    names = list(cols)
    return np.column_stack([cols[k] for k in names]), names


def _fit_spec(frame: pd.DataFrame, sym: str, spec_name: str, *, draws=3000, seed=SEED,
              extra_symbols: tuple[str, ...] = ()):
    """One Beta-Binomial ANCOVA. ``spec_name`` selects the word-reading term."""
    conf = TARGETS[sym]
    n_items = conf["n_items"]
    y = frame[f"{sym}_post"].to_numpy(int)
    X, cov_names = _covariate_matrix(frame, sym, n_items, extra_symbols)
    wr = frame["wr_pre"].to_numpy(float)
    wr_sd = float(np.std(wr)) or 1.0
    subj = pd.Categorical(frame["subject_id"]).codes
    n_subj = int(subj.max()) + 1

    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0.0, 1.5)
        beta = pm.Normal("beta", 0.0, 0.3, shape=X.shape[1])
        sigma_u = pm.HalfNormal("sigma_subject", 0.5)
        u_raw = pm.Normal("u_raw", 0.0, 1.0, shape=n_subj)
        kappa = pm.HalfNormal("kappa", 50.0)

        eta = alpha + pm.math.dot(X, beta) + (sigma_u * u_raw)[subj]

        if spec_name == "linear":
            b_w = pm.Normal("b_wr", 0.0, 0.3)
            eta = eta + b_w * (wr / wr_sd)
        elif spec_name == "log":
            lw = np.log1p(wr)
            b_w = pm.Normal("b_wr", 0.0, 0.3)
            eta = eta + b_w * ((lw - lw.mean()) / (lw.std() or 1.0))
        elif spec_name == "hinge":
            # Flat below theta, straight line above. theta is free on the raw word
            # scale so the answer is reported in words. The hinge is softened
            # (softplus with a 2-word scale) purely so NUTS sees a smooth gradient.
            theta = pm.Uniform("theta_words", 0.0, 40.0)
            b_w = pm.Normal("b_wr", 0.0, 0.3)
            soft = 2.0
            basis = soft * pm.math.log1pexp((wr - theta) / soft)
            eta = eta + b_w * (basis / wr_sd)
        elif spec_name != "null":
            raise ValueError(spec_name)

        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        pm.BetaBinomial("y", alpha=p * kappa, beta=(1 - p) * kappa, n=n_items,
                        observed=y)
        trace = pm.sample(draws=draws, tune=draws, chains=4, cores=4,
                          target_accept=0.95, nuts_sampler="nutpie",
                          random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": True})
    diag = az.summary(trace, var_names=[rv.name for rv in model.free_RVs]).astype(float)
    return trace, {
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess": float(diag["ess_bulk"].min()),
        "n_div": int(trace.sample_stats.diverging.values.sum()),
        "n_rows": len(frame), "n_children": n_subj, "cov_names": ",".join(cov_names),
    }


def _post(trace, name: str) -> dict[str, float]:
    x = trace.posterior[name].stack(sample=("chain", "draw")).values.ravel()
    lo, hi = (1 - CI) / 2, 1 - (1 - CI) / 2
    return {"median": float(np.median(x)), "lo": float(np.quantile(x, lo)),
            "hi": float(np.quantile(x, hi)), "lo50": float(np.quantile(x, 0.25)),
            "hi50": float(np.quantile(x, 0.75)), "p_pos": float((x > 0).mean())}


def run_p3(df: pd.DataFrame, out: Path, draws: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits, comparisons = [], []
    for sym, conf in TARGETS.items():
        frame = analysis_frame(df, sym, conf["room_at_or_below"])
        traces = {}
        for spec_name in ("null", "linear", "log", "hinge"):
            trace, diag = _fit_spec(frame, sym, spec_name, draws=draws)
            traces[spec_name] = trace
            row = {"target": sym, "label": conf["label"], "spec": spec_name, **diag}
            if spec_name != "null":
                row.update({f"b_wr_{k}": v for k, v in _post(trace, "b_wr").items()})
            if spec_name == "hinge":
                row.update({f"theta_{k}": v for k, v in
                            _post(trace, "theta_words").items()})
                # How much of the posterior sits below the smallest observed
                # non-zero word-reading score - i.e. "no threshold at all".
                th = trace.posterior["theta_words"].values.ravel()
                row["theta_p_below_1_word"] = float((th < 1).mean())
                row["theta_p_above_25_words"] = float((th > 25).mean())
            fits.append(row)
        # ArviZ 1.x: `compare` is PSIS-LOO by default and no longer takes `ic`.
        cmp_df = az.compare(traces, method="stacking")
        cmp_df = cmp_df.reset_index().rename(columns={"index": "spec"})
        cmp_df.insert(0, "target", sym)
        comparisons.append(cmp_df)
    fit_table = pd.DataFrame(fits)
    cmp_table = pd.concat(comparisons, ignore_index=True)
    fit_table.to_csv(out / "p3_specification_fits.csv", index=False)
    cmp_table.to_csv(out / "p3_loo_comparison.csv", index=False)
    return fit_table, cmp_table


# ---------------------------------------------------------------------------
# P4 - is it word reading, or the letter-sound knowledge that goes with it?
# ---------------------------------------------------------------------------


def run_p4(df: pd.DataFrame, out: Path, draws: int) -> pd.DataFrame:
    """Refit the blending and nonword specs with period-start LETTER SOUNDS added.

    Word-reading level and letter-sound knowledge are strongly coupled, and in the
    DAG letter sounds are the parent of nonword reading, not word reading. If the
    word-reading term collapses once letter sounds are held fixed, then any
    apparent "minimum reading level" is a minimum LETTER-SOUND level wearing a
    word-reading costume. Reported for both the linear and hinge specifications.

    This is a descriptive discrimination test, not an identification claim: letter
    sounds at period start are themselves a consequence of earlier reading, so
    conditioning on them can block part of the very path being asked about.
    """
    rows = []
    for sym in ("B", "N"):
        conf = TARGETS[sym]
        frame = analysis_frame(df, sym, conf["room_at_or_below"])
        frame = frame.dropna(subset=["L_pre"]) if "L_pre" in frame else frame
        for spec_name in ("linear", "hinge"):
            for extra, tag in (((), "no_LS"), (("L",), "with_LS")):
                trace, diag = _fit_spec(frame, sym, spec_name, draws=draws,
                                        extra_symbols=extra)
                row = {"target": sym, "label": conf["label"], "spec": spec_name,
                       "adjustment": tag, **diag}
                row.update({f"b_wr_{k}": v for k, v in _post(trace, "b_wr").items()})
                if spec_name == "hinge":
                    row.update({f"theta_{k}": v for k, v in
                                _post(trace, "theta_words").items()})
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "p4_letter_sound_discrimination.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Figures - one figure per file (PNG + SVG + the plotted series as CSV)
# ---------------------------------------------------------------------------

FIG_DIR = Path("notes/assets")
FIG_STEM = "202607241900"


def _save(fig, name: str, data: pd.DataFrame, out: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / f"{FIG_STEM}-{name}"
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    data.to_csv(out / f"figure_data_{name.replace('-', '_')}.csv", index=False)
    print(f"  wrote {base}.png / .svg")


def run_figures(df: pd.DataFrame, out: Path, draws: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import expit
    from scipy.stats import beta as beta_dist

    lo_q, hi_q = (1 - CI) / 2, 1 - (1 - CI) / 2

    # --- fig 1: the fitted nonword curve, the note's headline ----------------
    frame_n = analysis_frame(df, "N", TARGETS["N"]["room_at_or_below"])
    trace, _ = _fit_spec(frame_n, "N", "log", draws=draws)
    post = trace.posterior
    a = post["alpha"].stack(s=("chain", "draw")).values
    b = post["b_wr"].stack(s=("chain", "draw")).values
    beta_c = post["beta"].stack(s=("chain", "draw")).values
    X, _ = _covariate_matrix(frame_n, "N", 6)
    lw = np.log1p(frame_n["wr_pre"].to_numpy(float))
    base = a + beta_c.T @ X.mean(axis=0)

    grid = np.linspace(0, 45, 120)
    curve = []
    for w in grid:
        p = expit(base + b * ((np.log1p(w) - lw.mean()) / (lw.std() or 1.0)))
        items = p * 6
        curve.append((w, np.median(items), np.quantile(items, lo_q),
                      np.quantile(items, hi_q)))
    curve_df = pd.DataFrame(curve, columns=["prior_words", "median", "lo", "hi"])

    observed = (frame_n.assign(band=frame_n["band"])
                .groupby("band")
                .agg(prior_words=("wr_pre", "median"), observed=("N_post", "mean"),
                     n_rows=("N_post", "size"))
                .reset_index())

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(curve_df["prior_words"], curve_df["lo"], curve_df["hi"],
                    alpha=0.22, color="#2c6fbb", linewidth=0)
    ax.plot(curve_df["prior_words"], curve_df["median"], color="#2c6fbb", lw=2)
    ax.scatter(observed["prior_words"], observed["observed"], color="#b3452c",
               zorder=5, s=34)
    for _, r in observed.iterrows():
        ax.annotate(f"{r['band']} (n={int(r['n_rows'])})",
                    (r["prior_words"], r["observed"]), textcoords="offset points",
                    xytext=(6, -11), fontsize=8, color="#b3452c")
    ax.set_xlabel("Words read at the start of the period (EWRSWR)")
    ax.set_ylabel("Nonword reading at the end of the period (of 6)")
    ax.set_title("Nonword decoding rises fastest over the first few words read",
                 fontsize=11)
    ax.margins(x=0.02)
    fig.text(0.5, -0.04, "Fitted log specification, median and 89% interval, with "
             "covariates held at their means. Points are RAW band means, not "
             "adjusted for the period-start nonword score, so they sit above the "
             "curve where a band also started higher. Adjusted association, not "
             "causal.", ha="center", fontsize=7.5, color="#555555")
    _save(fig, "fig1-nonword-curve", curve_df.assign(kind="fitted"), out)
    plt.close(fig)

    # --- fig 2: floor exit by band ------------------------------------------
    frame_all = analysis_frame(df, "N", None)
    at_floor = frame_all[frame_all["N_pre"] == 0]
    rows = []
    for name, _lo, _hi in BANDS:
        b_rows = at_floor[at_floor["band"] == name]
        if b_rows.empty:
            continue
        k = int((b_rows["N_post"] > 0).sum())
        n = len(b_rows)
        # Jeffreys interval; rows are not independent (up to three per child), so
        # this is an optimistic width and is labelled as such in the note.
        rows.append({"band": name, "n_rows": n, "n_children": b_rows["subject_id"].nunique(),
                     "n_exited": k, "pct": 100 * k / n,
                     "lo": 100 * beta_dist.ppf(lo_q, k + 0.5, n - k + 0.5),
                     "hi": 100 * beta_dist.ppf(hi_q, k + 0.5, n - k + 0.5)})
    exit_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    xs = np.arange(len(exit_df))
    # A band where every child exited has a point estimate of 100% sitting just
    # outside its own Jeffreys upper limit; clip so the bar is drawn at zero
    # length rather than negative.
    ax.errorbar(xs, exit_df["pct"],
                yerr=[np.clip(exit_df["pct"] - exit_df["lo"], 0, None),
                      np.clip(exit_df["hi"] - exit_df["pct"], 0, None)],
                fmt="o", color="#2c6fbb", capsize=4)
    for x, r in zip(xs, exit_df.itertuples(), strict=True):
        ax.annotate(f"{r.n_exited}/{r.n_rows}", (x, r.pct), textcoords="offset points",
                    xytext=(9, 3), fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(exit_df["band"])
    ax.set_xlabel("Words read at the start of the period")
    ax.set_ylabel("Off the nonword floor by the end (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Leaving the nonword floor, by period-start word reading", fontsize=11)
    fig.text(0.5, -0.05, "Transitions starting at 0 of 6 nonwords. Jeffreys 89% "
             "intervals; rows are child-periods, so the widths are optimistic.",
             ha="center", fontsize=7.5, color="#555555")
    _save(fig, "fig2-nonword-floor-exit", exit_df, out)
    plt.close(fig)

    # --- fig 3: letter-sound gains, including the zero-word readers ----------
    frame_l = analysis_frame(df, "L", TARGETS["L"]["room_at_or_below"])
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    strip = []
    for i, (name, _lo, _hi) in enumerate(BANDS):
        vals = frame_l[frame_l["band"] == name]["L_gain"].to_numpy(float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, vals.size)
        colour = "#b3452c" if name == "0 words" else "#5b6b7a"
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=22, alpha=0.65,
                   color=colour, edgecolor="none")
        ax.hlines(np.median(vals), i - 0.28, i + 0.28, color="#1a1a1a", lw=2)
        strip.extend({"band": name, "gain": float(v)} for v in vals)
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([b[0] for b in BANDS])
    ax.set_xlabel("Words read at the start of the period")
    ax.set_ylabel("Letter sounds gained over the period (of 32)")
    ax.set_title("Children who read no words still learn letter sounds", fontsize=11)
    fig.text(0.5, -0.05, "Transitions with headroom (period-start letter sounds "
             "≤ 24 of 32). Bars are band medians.", ha="center", fontsize=7.5,
             color="#555555")
    _save(fig, "fig3-letter-sound-gains", pd.DataFrame(strip), out)
    plt.close(fig)

    # --- fig 4: the breakpoint posteriors are flat ---------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    theta_rows = []
    colours = {"L": "#2c6fbb", "B": "#4d9a6a", "N": "#b3452c"}
    for sym, conf in TARGETS.items():
        frame = analysis_frame(df, sym, conf["room_at_or_below"])
        tr, _ = _fit_spec(frame, sym, "hinge", draws=draws)
        th = tr.posterior["theta_words"].values.ravel()
        ax.hist(th, bins=40, range=(0, 40), density=True, histtype="step", lw=2,
                color=colours[sym], label=f"{conf['label']} (median "
                                          f"{np.median(th):.1f} words)")
        theta_rows.extend({"target": sym, "theta_words": float(t)}
                          for t in th[:: max(1, len(th) // 2000)])
    ax.set_xlabel("Fitted breakpoint θ (words read)")
    ax.set_ylabel("Posterior density")
    ax.set_title("The fitted breakpoint is either at zero or nowhere", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    fig.text(0.5, -0.07, "Free-breakpoint hinge specification, prior Uniform(0, 40) "
             "words. Two distinct readings, neither of them a threshold: nonword "
             "reading piles up against zero (the rise starts at the first word), "
             "while letter sounds and blending stay diffuse across the whole range "
             "(no breakpoint the data can locate).", ha="center", fontsize=7.5,
             color="#555555")
    _save(fig, "fig4-breakpoint-posteriors", pd.DataFrame(theta_rows), out)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("output/notes/202607241900-wr-threshold"))
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--figures", action="store_true",
                    help="write the note's figures (refits the specs they need)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = transitions()
    print(f"transitions: {len(df)} rows, {df['subject_id'].nunique()} children\n")

    p1 = run_p1(df, args.out)
    print("P1 - progress by period-start word-reading band (room-restricted)")
    show = p1[p1["restriction"] == "room"]
    print(show[["target", "band", "n_rows", "median_pre", "median_gain",
                "pct_any_gain", "mean_resid_gain", "cliffs_delta_vs_zero_band",
                "delta_lo", "delta_hi"]].to_string(index=False))

    existence, exits = run_p2(df, args.out)
    print("\nP2 - do zero-word readers progress at all?")
    print(existence.to_string(index=False))
    print("\nP2 - floor exit by band")
    print(exits.to_string(index=False))

    if not args.skip_models:
        fit_table, cmp_table = run_p3(df, args.out, args.draws)
        print("\nP3 - specification fits")
        print(fit_table.to_string(index=False))
        print("\nP3 - PSIS-LOO comparison")
        print(cmp_table.to_string(index=False))

        p4 = run_p4(df, args.out, args.draws)
        print("\nP4 - word reading, or the letter sounds that go with it?")
        print(p4[["target", "spec", "adjustment", "n_rows", "b_wr_median",
                  "b_wr_lo", "b_wr_hi", "b_wr_p_pos", "max_rhat", "n_div"]]
              .to_string(index=False))

    if args.figures:
        print("\nFigures")
        run_figures(df, args.out, args.draws)

    print(f"\nCSVs written to {args.out}")


if __name__ == "__main__":
    main()
