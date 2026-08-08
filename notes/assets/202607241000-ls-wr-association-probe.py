# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Letter-sound knowledge (LS) -> word-reading (WR) *level* associations, exploratory probe.

Supports ``notes/202607241000-findings-letter-sounds-word-reading-association.md``.
Four questions, all **adjusted associations** (latent general ability is unblockable;
nothing here is causal — only the ITT arm is randomised):

Q1  How strongly is LS associated with the WR *level*, holding age, hearing and
    non-verbal ability (WPPSI block design) fixed?
Q2  Does that association survive additionally holding nonword reading (NW) fixed?
Q3  Among children alike on LS (+ the Q1 adjusters), what else predicts WR — i.e.
    what distinguishes children who know their letter sounds but read few words?
Q4  How does vocabulary relate to word LEARNING (gains), irrespective of LS? Both
    readings of "word learning" are fitted: learning to READ words (outcome W) and
    learning new WORDS (the taught sets TR / TE).

Q1-Q3 reuse the registered ``concurrent`` family's factory
(``factories.build_concurrent_model``): a per-wave, between-child Beta-Binomial
regression of the WR item count on standardised same-wave logits, one row per child,
no own baseline, no child random intercept. Q4 is a *gains* question, so it reuses the
``gain_factors`` factory instead (period-transition ANCOVA over the three transitions
with a child random intercept). Sampling is ``rep-lite``-equivalent (4 chains x
2000-4000 draws, ``target_accept=0.95``, nutpie).

These are **exploratory scratch fits**, not registered models: they publish no
``config.json`` / ``diagnostics_summary.json`` / report, and they bypass the
production convergence gate (R-hat, ESS, BFMI and divergences are checked inline and
recorded in the note instead). Promote to ``lrp_rli_ca_0NN`` / ``lrp_rli_gf_0NN`` modules before citing
anywhere outside the note.

Run with the conda env interpreter from the repo root; writes CSVs to ``--out``.
Use ``--section q3-partials`` to rerun only the timing-sensitive Q3 partial fits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

from language_reading_predictors.data_utils import load_data
from language_reading_predictors.statistical_models import factories as F
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    load_and_prepare,
    logit_safe,
    standardise,
)

CI = 0.89  # house standard (notes/202607172359-credible-interval-standard.md)
BASE_COV = ["blocks", "hs"]  # non-verbal ability + hearing; age enters via include_age


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _summarise(trace, name: str) -> dict[str, float]:
    """Posterior median, equal-tailed 89% interval, inner 50% band and P(>0)."""
    x = trace.posterior[name].stack(sample=("chain", "draw")).values.ravel()
    lo_q, hi_q = (1 - CI) / 2, 1 - (1 - CI) / 2
    return {
        "median": float(np.median(x)),
        "lo": float(np.quantile(x, lo_q)),
        "hi": float(np.quantile(x, hi_q)),
        "lo50": float(np.quantile(x, 0.25)),
        "hi50": float(np.quantile(x, 0.75)),
        "p_pos": float((x > 0).mean()),
    }


def _fit(sub, predictors, covariates, *, age=True, group=False, draws=4000, seed=20260724):
    built = F.build_concurrent_model(
        sub,
        outcome_symbol="W",
        predictor_symbols=predictors,
        covariates=covariates,
        include_age=age,
        include_group=group,
        predictor_slope_sigma=0.3,
    )
    with built.model:
        trace = pm.sample(
            draws=draws,
            tune=draws // 2,
            chains=4,
            cores=4,
            target_accept=0.95,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=seed,
            progressbar=False,
        )
    summary = az.summary(
        trace, var_names=[rv.name for rv in built.model.free_RVs]
    ).astype(float)
    diag = {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess": float(summary["ess_bulk"].min()),
        "n_div": int(trace.sample_stats.diverging.values.sum()),
    }
    return trace, built, diag


def _terms(sub, symbols):
    """``ConcurrentTerm`` list so a coefficient can be pushed to the items scale."""
    out = []
    for sym in symbols:
        m = MEASURES[sym]
        vals = np.asarray(sub.post_counts[sym], dtype=float)
        _z, scaler = standardise(logit_safe(vals, m.n_trials))
        out.append(
            R.ConcurrentTerm(
                label=sym,
                coef=f"beta_{sym}",
                sd_logit=float(scaler.sd),
                n_items=m.n_trials,
                mean_items=float(np.nanmean(vals)),
                k_items=max(1, round(m.n_trials / 10)),
            )
        )
    return out


def _waves(prepared):
    """Yield ``(timepoint, single-wave subset with an observed WR outcome)``."""
    for w in sorted({int(p) for p in np.unique(prepared.phase)}):
        sub = _subset_prepared(prepared, prepared.phase == w)
        yield w + 1, _subset_prepared(sub, ~np.isnan(sub.post_counts["W"]))


# ---------------------------------------------------------------------------
# Q1 / Q2 — the LS slope under the requested adjustment sets
# ---------------------------------------------------------------------------

Q12_SPECS = {
    # label -> (predictors, covariates, include_age, include_group)
    "Q0 unadjusted": (["L"], [], False, False),
    "Q1 +age+hearing+ability": (["L"], BASE_COV, True, False),
    "Q1g +arm nuisance": (["L"], BASE_COV, True, True),
    "Q2 +nonword reading": (["L", "N"], BASE_COV, True, False),
}


def run_q1_q2(prepared, out: Path) -> pd.DataFrame:
    rows: list[dict] = []
    draws_by_key: dict[str, np.ndarray] = {}
    for tp, sub in _waves(prepared):
        for label, (preds, cov, age, group) in Q12_SPECS.items():
            trace, built, diag = _fit(sub, preds, cov, age=age, group=group, seed=20260724 + tp)
            marginals = R.concurrent_marginals(
                trace, terms=_terms(built.prepared, preds),
                n_trials=MEASURES["W"].n_trials, ci_prob=CI,
            )
            reported = [(s, f"beta_{s}") for s in preds]
            reported += [(c, f"gamma_{c}") for c in cov]
            if age:
                reported.append(("age", "beta_age"))
            for term, var in reported:
                if var not in trace.posterior:
                    continue
                row = {"timepoint": tp, "model": label, "term": term,
                       "n": built.prepared.n_obs, **_summarise(trace, var), **diag}
                sd_row = marginals[(marginals.term == term) & (marginals.scale == "+1 SD")]
                if len(sd_row) == 1:
                    row |= {
                        "items_median": float(sd_row.items_median.iloc[0]),
                        "items_lo": float(sd_row.items_lo.iloc[0]),
                        "items_hi": float(sd_row.items_hi.iloc[0]),
                    }
                rows.append(row)
                draws_by_key[f"t{tp}|{label}|{term}"] = (
                    trace.posterior[var].stack(sample=("chain", "draw")).values.ravel()
                )
            print(f"t{tp} {label}: rhat={diag['max_rhat']:.3f} "
                  f"ess={diag['min_ess']:.0f} div={diag['n_div']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "q1_q2_ls_wr_adjusted.csv", index=False)

    # Q2 attenuation. The two fits are SEPARATE, so this is a product-of-marginals
    # sensitivity under working independence — NOT an identified posterior contrast
    # (same caveat as the Tier-1 Delta, notes/202607172358-findings-decoding-specificity.md).
    rng = np.random.default_rng(7)
    att = []
    for tp, _ in _waves(prepared):
        a = draws_by_key[f"t{tp}|Q1 +age+hearing+ability|L"]
        b = draws_by_key[f"t{tp}|Q2 +nonword reading|L"]
        n = min(len(a), len(b))
        a, b = rng.permutation(a)[:n], rng.permutation(b)[:n]
        share = b / a
        att.append({
            "timepoint": tp,
            "ls_slope_q1": float(np.median(a)),
            "ls_slope_q2": float(np.median(b)),
            "share_retained": float(np.median(share)),
            "share_lo": float(np.quantile(share, 0.055)),
            "share_hi": float(np.quantile(share, 0.945)),
            "p_residual_positive": float((b > 0).mean()),
        })
    att_df = pd.DataFrame(att)
    att_df.to_csv(out / "q2_nonword_attenuation.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Q3a — partial associations with WR, holding LS + the Q1 adjusters fixed
# ---------------------------------------------------------------------------

CANDIDATE_MEASURES = ["N", "B", "R", "E", "T", "F"]  # same-wave measured skills
CANDIDATE_COVARIATES = ["erbto", "deapp_c"]  # same-wave phonological memory + speech


def run_q3_partials(out: Path) -> pd.DataFrame:
    prepared = load_and_prepare(
        phase_mode="levels",
        outcomes=("W", "L", "N", "B", "R", "E", "T", "F"),
        # Block design is t1-only and hearing is time-invariant; phonological
        # memory and speech are repeatedly measured states and must come from
        # each level row rather than being broadcast from t1 (#421 closeout).
        baseline_covariates=("blocks", "hs"),
        post_covariates=tuple(CANDIDATE_COVARIATES),
        pre_required=(),
    )
    rows: list[dict] = []
    for tp, sub in _waves(prepared):
        for sym in CANDIDATE_MEASURES:
            trace, built, diag = _fit(sub, ["L", sym], BASE_COV, draws=3000, seed=424242)
            for term, var in (("L", "beta_L"), (sym, f"beta_{sym}")):
                rows.append({"timepoint": tp, "candidate": sym, "term": term,
                             "n": built.prepared.n_obs, **_summarise(trace, var), **diag})
        for cov in CANDIDATE_COVARIATES:
            trace, built, diag = _fit(sub, ["L"], BASE_COV + [cov], draws=3000, seed=424242)
            for term, var in (("L", "beta_L"), (cov, f"gamma_{cov}")):
                rows.append({"timepoint": tp, "candidate": cov, "term": term,
                             "n": built.prepared.n_obs, **_summarise(trace, var), **diag})
        # All candidates at once. Heavily collinear at n ~ 51 and shrunk by the
        # Normal(0, 0.3) slope prior — read under the Table-2 fallacy, never as a
        # ranking of importance.
        trace, built, diag = _fit(
            sub, ["L", "N", "B", "R", "E"], BASE_COV + CANDIDATE_COVARIATES,
            draws=3000, seed=424242,
        )
        for var in [v for v in trace.posterior.data_vars
                    if v.startswith(("beta_", "gamma_")) and v != "beta_group_nuisance"]:
            rows.append({"timepoint": tp, "candidate": "JOINT",
                         "term": var.split("_", 1)[1], "n": built.prepared.n_obs,
                         **_summarise(trace, var), **diag})
        print(f"t{tp} Q3 partials done (div={diag['n_div']})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "q3_partial_associations.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Q3b — the WR-given-LS residual and the discrepancy-group profile
# ---------------------------------------------------------------------------

PROFILE_VARS = [
    "ewrswr", "yarclet", "nonword", "blending", "spphon", "erbto", "erbnw", "erbword",
    "deapp_c", "rowpvt", "eowpvt", "trog", "celf", "blocks", "age", "hearing_c", "earinf",
]
# Session dose is not part of the subgroup profile (it is not a child trait) but the
# note quotes its residual correlation, so it joins the correlation sweep. It is
# recorded at t1-t3 only — the t4 column is empty by construction, not by chance.
CORRELATION_EXTRA = ["attend"]


def run_q3_profile(prepared, out: Path) -> pd.DataFrame:
    """Residualise WR on LS + age + hearing + ability, then profile the low residuals.

    Returns the per-wave residual frame, which ``run_q3_distribution`` reuses to rank
    the counter-cases on a continuous scale rather than by median-split membership.
    """
    frames = []
    diagnostics: list[dict] = []
    for tp, sub in _waves(prepared):
        trace, built, diag = _fit(sub, ["L"], BASE_COV, draws=2000, seed=99)
        diagnostics.append({"timepoint": tp, **diag})
        eta = (
            trace.posterior["eta"]
            .stack(sample=("chain", "draw"))
            .transpose("obs_id", "sample")
            .values
        )
        observed = built.prepared.post_counts["W"]
        expected = MEASURES["W"].n_trials * expit(eta).mean(axis=1)
        frames.append(pd.DataFrame({
            "subject_id": built.prepared.subject_ids, "time": tp,
            "wr_observed": observed, "wr_expected": expected,
            "resid": observed - expected,
        }))
    residuals = pd.concat(frames)
    residuals.to_csv(out / "q3_wr_given_ls_residuals.csv", index=False)
    # These four residualising fits publish no coefficient table, but the note quotes a
    # convergence claim over EVERY fit in the probe, so record their diagnostics too.
    pd.DataFrame(diagnostics).to_csv(out / "q3_residual_fit_diagnostics.csv", index=False)

    merged = load_data().merge(residuals, on=["subject_id", "time"], how="inner")

    # Stability of the discrepancy across waves.
    stability = residuals.pivot(index="subject_id", columns="time", values="resid").corr()
    stability.to_csv(out / "q3_residual_stability.csv")

    # Whole-cohort Spearman correlation of the residual with each candidate, per wave.
    candidates = [c for c in PROFILE_VARS if c not in {"ewrswr", "yarclet"}] + CORRELATION_EXTRA
    corr = {
        f"t{t}": {
            c: (
                merged.loc[merged.time == t, ["resid", c]].dropna()
                .corr(method="spearman").iloc[0, 1]
                if merged.loc[merged.time == t, c].notna().sum() >= 3
                else np.nan  # e.g. ``attend``, which is not recorded at t4
            )
            for c in candidates
        }
        for t in (1, 2, 3, 4)
    }
    pd.DataFrame(corr).to_csv(out / "q3_residual_correlations.csv")

    # t4 discrepancy profile: high-LS children split at that group's median WR.
    t4 = merged[merged.time == 4]
    high_ls = t4[t4.yarclet >= 24].copy()
    cut = high_ls.ewrswr.median()
    high_ls["band"] = np.where(high_ls.ewrswr <= cut, "lower_WR", "higher_WR")
    rows = []
    for var in PROFILE_VARS:
        lo = high_ls.loc[high_ls.band == "lower_WR", var].dropna().astype(float)
        hi = high_ls.loc[high_ls.band == "higher_WR", var].dropna().astype(float)
        if len(lo) < 3 or len(hi) < 3:
            continue
        pooled_sd = np.sqrt(
            ((len(lo) - 1) * lo.var(ddof=1) + (len(hi) - 1) * hi.var(ddof=1))
            / (len(lo) + len(hi) - 2)
        )
        rows.append({
            "var": var, "n_lower": len(lo), "n_higher": len(hi),
            "median_lower": lo.median(), "median_higher": hi.median(),
            "mean_lower": lo.mean(), "mean_higher": hi.mean(),
            "cohens_d": (lo.mean() - hi.mean()) / pooled_sd if pooled_sd > 0 else np.nan,
        })
    pd.DataFrame(rows).sort_values("cohens_d").to_csv(
        out / "q3_high_ls_profile_t4.csv", index=False)
    return residuals


# ---------------------------------------------------------------------------
# Q3c — the raw joint distribution at t4, and the counter-cases
#
# No model: the observed cross-tabulation of word reading against letter sounds,
# the LS profile of each WR quartile / half, and the children who read at or above
# the median while knowing fewer letter sounds than the median. Deliberately
# model-free — the point is to show what the cut-point choice does to the answer.
# ---------------------------------------------------------------------------


def run_q3_distribution(out: Path, residuals: pd.DataFrame | None = None) -> None:
    raw = load_data()
    t4 = raw[raw.time == 4].dropna(subset=["ewrswr", "yarclet"]).copy()
    if residuals is not None:
        t4 = t4.merge(residuals[residuals.time == 4][["subject_id", "wr_expected", "resid"]],
                      on="subject_id", how="left")
    n = len(t4)
    med_wr, med_ls = t4.ewrswr.median(), t4.yarclet.median()

    # Letter-sound profile of each word-reading quartile and half. Ranked
    # assignment, so the children tied at the quartile boundary split arbitrarily —
    # recorded because that tie is what drives the "lowest quartile" ambiguity.
    t4["quartile"] = pd.qcut(t4.ewrswr.rank(method="first"), 4,
                             labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    t4["half"] = pd.qcut(t4.ewrswr.rank(method="first"), 2,
                         labels=["bottom 50%", "top 50%"])
    frames = []
    for col in ("quartile", "half"):
        g = t4.groupby(col, observed=True).agg(
            n=("ewrswr", "size"), wr_min=("ewrswr", "min"), wr_max=("ewrswr", "max"),
            wr_median=("ewrswr", "median"), ls_median=("yarclet", "median"),
            ls_q1=("yarclet", lambda s: s.quantile(0.25)),
            ls_q3=("yarclet", lambda s: s.quantile(0.75)),
            ls_min=("yarclet", "min"), ls_max=("yarclet", "max"),
        )
        for thr in (20, 24, 28):
            g[f"pct_ge_{thr}"] = (
                t4.groupby(col, observed=True).yarclet.apply(lambda s: 100 * (s >= thr).mean())
            )
        frames.append(g.reset_index().rename(columns={col: "group"}).assign(split=col))
    pd.concat(frames).to_csv(out / "q3_ls_by_wr_band_t4.csv", index=False)

    # The 2x2 at the two medians, and the counter-cases inside it.
    pd.crosstab(np.where(t4.yarclet >= med_ls, f"LS >= {med_ls:.0f}", f"LS < {med_ls:.0f}"),
                np.where(t4.ewrswr >= med_wr, f"WR >= {med_wr:.0f}", f"WR < {med_wr:.0f}")
                ).to_csv(out / "q3_median_crosstab_t4.csv")

    counter = t4[(t4.ewrswr >= med_wr) & (t4.yarclet < med_ls)].copy()
    counter["ls_below_median"] = med_ls - counter.yarclet
    counter["wr_above_median"] = counter.ewrswr - med_wr
    if "resid" in counter:
        counter["resid_rank"] = t4.resid.rank(ascending=False)[counter.index]
    cols = ["subject_id", "ewrswr", "yarclet", "ls_below_median", "wr_above_median",
            "nonword", "blending", "spphon", "erbto", "erbnw", "deapp_c", "rowpvt",
            "eowpvt", "trog", "celf", "blocks", "age", "group"]
    cols += [c for c in ("wr_expected", "resid", "resid_rank") if c in counter]
    counter[cols].sort_values("ewrswr", ascending=False).to_csv(
        out / "q3_counter_cases_t4.csv", index=False)

    # Does the counter-case CELL hold the same children wave to wave? (It does not —
    # median-split membership churns even though the continuous residual is stable.)
    membership = {}
    for t in (1, 2, 3, 4):
        g = raw[raw.time == t].dropna(subset=["ewrswr", "yarclet"])
        membership[t] = set(
            g[(g.ewrswr >= g.ewrswr.median()) & (g.yarclet < g.yarclet.median())].subject_id
        )
    ever = sorted(set().union(*membership.values()))
    pd.DataFrame([
        {"subject_id": s, **{f"t{t}": int(s in membership[t]) for t in (1, 2, 3, 4)},
         "n_waves": sum(s in membership[t] for t in (1, 2, 3, 4))}
        for s in ever
    ]).sort_values("n_waves", ascending=False).to_csv(
        out / "q3_counter_case_stability.csv", index=False)
    print(f"t4: n={n}, median WR={med_wr:.0f}, median LS={med_ls:.0f}; "
          f"{len(counter)} counter-cases; {len(ever)} children in the cell at least once")


# ---------------------------------------------------------------------------
# Q4 — vocabulary and word LEARNING (gains), with and without letter sounds
#
# "Word learning" is ambiguous, so both readings are fitted:
#   A  learning to READ words -> outcome W
#   B  learning new WORDS     -> outcomes TR / TE (the bespoke taught-word sets)
# Each is fitted twice, with skill adjusters {R, E, L} and {R, E}; the contrast is
# exactly "irrespective of letter-sound knowledge".
#
# These use the registered ``gain_factors`` factory: a period-transition ANCOVA
# (post given the period's own pre) stacked over the three transitions, with a child
# random intercept. Only ``beta_trt`` is causal; every gamma is an adjusted
# association and the random intercept is a partial, shrunken stand-in for
# between-child heterogeneity, NOT a control for latent general ability.
# ---------------------------------------------------------------------------

Q4_SPECS = [
    ("A  W  learning to read words", "W", ("R", "E", "L")),
    ("A' W  without LS", "W", ("R", "E")),
    ("B  TR learning taught receptive words", "TR", ("R", "E", "L")),
    ("B' TR without LS", "TR", ("R", "E")),
    ("B  TE learning taught expressive words", "TE", ("R", "E", "L")),
    ("B' TE without LS", "TE", ("R", "E")),
]


def run_q4_word_learning(out: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for label, outcome, skills in Q4_SPECS:
        prepared = load_and_prepare(
            phase_mode="all", outcomes=(outcome, *skills),
            baseline_covariates=("blocks",), post_covariates=("hs",),
        )
        adjust = tuple(c for c in ("hs", "hs_missing") if c in prepared.covariates)
        built = F.build_gain_factors_model(
            prepared, outcome_symbol=outcome, skill_symbols=skills,
            ability_covariate="blocks", adjust_for=adjust, interactions=(),
        )
        with built.model:
            trace = pm.sample(
                draws=4000, tune=2000, chains=4, cores=4, target_accept=0.95,
                nuts_sampler="nutpie", return_inferencedata=True,
                random_seed=20260724, progressbar=False,
            )
        summary = az.summary(
            trace, var_names=[rv.name for rv in built.model.free_RVs]
        ).astype(float)
        diag = {"max_rhat": float(summary["r_hat"].max()),
                "min_ess": float(summary["ess_bulk"].min()),
                "n_div": int(trace.sample_stats.diverging.values.sum())}
        for var in trace.posterior.data_vars:
            if not var.startswith(("gamma_", "beta_trt")) or "int_" in var:
                continue
            if trace.posterior[var].ndim > 2:  # skip vector / per-obs deterministics
                continue
            rows.append({"model": label, "outcome": outcome, "skills": "+".join(skills),
                         "term": var, "n_rows": built.prepared.n_obs,
                         **_summarise(trace, var), **diag})
        print(f"{label}: rhat={diag['max_rhat']:.3f} ess={diag['min_ess']:.0f} "
              f"div={diag['n_div']} rows={built.prepared.n_obs}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "q4_vocabulary_word_learning.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/notes/202607241000-ls-wr"))
    parser.add_argument(
        "--section",
        choices=("all", "q3-partials"),
        default="all",
        help="Run the full probe (default) or only the Q3 partial-association fits.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.section == "q3-partials":
        run_q3_partials(args.out)
        print(f"Wrote Q3 partial-association CSV to {args.out}")
        return

    prepared = load_and_prepare(
        phase_mode="levels", outcomes=("W", "L", "N"),
        baseline_covariates=("blocks", "hs"), pre_required=(),
    )
    run_q1_q2(prepared, args.out)
    run_q3_partials(args.out)
    residuals = run_q3_profile(prepared, args.out)
    run_q3_distribution(args.out, residuals=residuals)

    run_q4_word_learning(args.out)
    print(f"Wrote CSVs to {args.out}")


if __name__ == "__main__":
    main()
