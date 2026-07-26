# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Can item-difficulty non-exchangeability explain the negative word-reading age slope?

Companion analysis to
notes/202607261405-binomial-exchangeability-item-difficulty-review.md. The suite
finds a credibly negative linear-age precision term for word reading conditional on
own baseline (``gamma_A`` ~ -0.12 logit; adj-065 -0.26; the errors-in-variables fit
-0.34). The hypothesis under test: older children sit higher on the item ladder where
words are harder, so the slope is a measurement artefact. Two children with the same
baseline score face the same next words regardless of age, so the artefact can only
enter through unmodelled baseline curvature absorbed by age via the age-baseline
correlation. This script tests that channel twice:

Part 1 (real data, randomised t1->t2 window): regress W post on baseline + age + arm
with the baseline entered (a) linearly and (b) with hinge terms at 25/30 items (the
documented kink region). If the negative age slope is curvature leak, it should
attenuate under (b).

Part 2 (simulation under the null): Rasch-type 79-item ladders (kinked EWR/SWR-style,
smoothly graded, homogeneous), age correlated with baseline ability at the observed
level, and growth INDEPENDENT of age (true age effect = 0). The fitted age
coefficient measures how much spurious slope each ladder can generate.

Result recorded in the note: every ladder yields a POSITIVE mean age coefficient
under the null (errors-in-baseline makes age a proxy for true ability), so the
observed negative slope cannot be a difficulty-ladder artefact.

Usage:
    python scripts/age_artefact_check.py [--reps 1000] [--seed 20260726] [--data data/rli_data_long.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

N_ITEMS = 79


def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def elogit(k, n):
    k = np.asarray(k, float)
    return np.log((k + 0.5) / (n - k + 0.5))


def ols(y, cols):
    names = list(cols)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(cols[c], float) for c in names])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {n_: (bi, si) for n_, bi, si in zip(["const"] + names, b, se, strict=True)}


def show(res, keys):
    for k in keys:
        b, s = res[k]
        print(f"    {k:>16}: {b:+7.3f}  (se {s:.3f})")


def real_data_checks(data_path: str) -> tuple[int, float, float, float, float]:
    df = pd.read_csv(data_path)
    w = df.pivot_table(index="subject_id", columns="time", values="ewrswr")
    a = df.pivot_table(index="subject_id", columns="time", values="age")
    g = df.groupby("subject_id")["group"].first()

    d = pd.DataFrame({"W1": w[1], "W2": w[2], "age1": a[1], "G": 2 - g}).dropna()
    n = len(d)
    print(f"== Real data: W (ewrswr), t1->t2, n = {n}")
    print(f"   W1 mean {d.W1.mean():.2f} sd {d.W1.std():.2f}; W2 mean {d.W2.mean():.2f}; "
          f"mean gain {(d.W2 - d.W1).mean():.2f}")
    r_age_w1 = float(np.corrcoef(d.age1, d.W1)[0, 1])
    print(f"   corr(age1, W1) = {r_age_w1:+.3f}   corr(age1, W2-W1) = "
          f"{np.corrcoef(d.age1, d.W2 - d.W1)[0, 1]:+.3f}")

    bands = pd.cut(d.W1, [-0.5, 5, 15, 25, 80], labels=["0-5", "6-15", "16-25", ">25"])
    tab = d.assign(gain=d.W2 - d.W1, band=bands).groupby("band", observed=True).agg(
        n=("gain", "size"), mean_gain=("gain", "mean"), mean_age=("age1", "mean"))
    print("   gain and age by baseline band:")
    print("   " + tab.to_string().replace("\n", "\n   "))

    y_raw = np.asarray(d.W2, float)
    y_el = elogit(d.W2, N_ITEMS)
    h25 = np.clip(d.W1 - 25, 0, None)
    h30 = np.clip(d.W1 - 30, 0, None)

    print("\n  [raw scale] W2 ~ z(W1) + z(age) + G")
    show(ols(y_raw, {"zW1": z(d.W1), "zage": z(d.age1), "G": d.G}), ["zW1", "zage", "G"])
    print("  [raw scale] + hinge(W1-25) + hinge(W1-30)")
    show(ols(y_raw, {"zW1": z(d.W1), "h25": h25, "h30": h30, "zage": z(d.age1), "G": d.G}),
         ["zW1", "h25", "h30", "zage", "G"])
    print("  [empirical logit] elogit(W2) ~ z(W1) + z(age) + G   (~ITT structure)")
    show(ols(y_el, {"zW1": z(d.W1), "zage": z(d.age1), "G": d.G}), ["zW1", "zage", "G"])
    print("  [empirical logit] + hinge terms")
    show(ols(y_el, {"zW1": z(d.W1), "h25": h25, "h30": h30, "zage": z(d.age1), "G": d.G}),
         ["zW1", "h25", "h30", "zage", "G"])

    for col, nmax, label in [("rowpvt", 170, "R (ROWPVT, graded ladder)"),
                             ("b1extau", 24, "TE (taught words, no ladder)")]:
        p = df.pivot_table(index="subject_id", columns="time", values=col)
        dd = pd.DataFrame({"y1": p[1], "y2": p[2], "age1": a[1], "G": 2 - g}).dropna()
        r = ols(elogit(dd.y2, nmax), {"zpre": z(dd.y1), "zage": z(dd.age1), "G": dd.G})
        b_, s_ = r["zage"]
        print(f"  {label}: n={len(dd)}  empirical-logit age coef {b_:+.3f} (se {s_:.3f})")

    return n, float(d.W1.mean()), float(d.W1.std()), r_age_w1, float((d.W2 - d.W1).mean())


def simulate(n, w1_mean, w1_sd, r_age_w1, gain_obs, reps, rng):
    print("\n== Simulation: ladders with ZERO true age effect on growth "
          f"(n = {n}, reps = {reps})")

    ladders = {
        "kinked (EWR easy, SWR jump)": np.r_[np.linspace(-2.5, 0.5, 30), np.linspace(1.5, 5.0, 49)],
        "smooth graded": np.linspace(-2.5, 5.0, 79),
        "homogeneous items": np.full(N_ITEMS, 2.0),
    }

    def score(theta, b):
        p = 1 / (1 + np.exp(-(theta[:, None] - b[None, :])))
        return (rng.random(p.shape) < p).sum(axis=1)

    def calibrate(b):
        best = None
        for mu in np.linspace(-4, 2, 25):
            for sd in np.linspace(0.5, 2.5, 9):
                th = rng.normal(mu, sd, 4000)
                k = score(th, b)
                loss = (k.mean() - w1_mean) ** 2 + (k.std() - w1_sd) ** 2
                if best is None or loss < best[0]:
                    best = (loss, mu, sd)
        return best[1], best[2]

    for name, b_items in ladders.items():
        mu_t, sd_t = calibrate(b_items)
        d0 = None
        for cand in np.linspace(0.05, 1.5, 30):
            th = rng.normal(mu_t, sd_t, 4000)
            gg = score(th + cand, b_items).mean() - score(th, b_items).mean()
            if d0 is None or abs(gg - gain_obs) < d0[0]:
                d0 = (abs(gg - gain_obs), cand)
        delta0 = d0[1]

        coefs_lin, coefs_hinge, coefs_raw = [], [], []
        for _ in range(reps):
            th1 = rng.normal(mu_t, sd_t, n)
            th1z = (th1 - mu_t) / sd_t
            age = r_age_w1 * th1z + np.sqrt(1 - r_age_w1**2) * rng.normal(size=n)
            delta = rng.normal(delta0, 0.3, n)  # independent of age: true effect 0
            k1 = score(th1, b_items)
            k2 = score(th1 + delta, b_items)
            yl = elogit(k2, N_ITEMS)
            try:
                coefs_lin.append(ols(yl, {"zW1": z(k1), "zage": z(age)})["zage"][0])
                hh = np.clip(k1 - 25, 0, None)
                coefs_hinge.append(ols(yl, {"zW1": z(k1), "h25": hh, "zage": z(age)})["zage"][0])
                coefs_raw.append(ols(k2.astype(float), {"zW1": z(k1), "zage": z(age)})["zage"][0])
            except np.linalg.LinAlgError:
                continue
        cl, ch, cr = map(np.asarray, (coefs_lin, coefs_hinge, coefs_raw))
        print(f"  {name}:  theta~N({mu_t:.2f},{sd_t:.2f}), delta0={delta0:.2f}")
        print(f"    empirical-logit age coef, linear baseline : mean {cl.mean():+.4f}  "
              f"[2.5,97.5%: {np.percentile(cl, 2.5):+.3f}, {np.percentile(cl, 97.5):+.3f}]  "
              f"P(<=-0.12) = {(cl <= -0.12).mean():.2f}")
        print(f"    empirical-logit age coef, hinge baseline  : mean {ch.mean():+.4f}")
        print(f"    raw-scale age coef                        : mean {cr.mean():+.4f} items")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--data", default="data/rli_data_long.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, w1_mean, w1_sd, r_age_w1, gain_obs = real_data_checks(args.data)
    simulate(n, w1_mean, w1_sd, r_age_w1, gain_obs, args.reps, rng)


if __name__ == "__main__":
    main()
