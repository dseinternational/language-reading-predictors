# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RI-CLPM feasibility study (L<->W direction), per notes/202607172230-riclpm-direction-plan.md.

Standalone (scratchpad) builder + numpy simulator + recovery harness. If the gate
passes we port build_riclpm_model into the package as a real family.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from language_reading_predictors.statistical_models.likelihood import beta_binomial_from_logit
from language_reading_predictors.statistical_models import priors as _priors

EPS = 1e-6

# ----------------------------------------------------------------------------- design
def real_design():
    """(counts, mask, n_trials, arm_idx, age_std, mu_anchor[arm,wave,out]) for L,W over 4 waves."""
    from language_reading_predictors.statistical_models.preprocessing import load_wave_panel
    panel = load_wave_panel(outcomes=("L", "W"))
    OUT = ["L", "W"]
    counts = np.stack([np.nan_to_num(panel.counts[s], nan=0.0).astype(np.int64) for s in OUT], axis=2)
    mask = np.stack([panel.obs_mask[s] for s in OUT], axis=2)
    n_trials = np.array([panel.n_trials[s] for s in OUT], dtype=int)
    arm_idx = (np.asarray(panel.group) == 2).astype(int)  # 0=immediate,1=waitlist
    age = np.asarray(panel.age_std)
    T = counts.shape[1]
    # arm x wave x out logit cell means (observed), for realistic simulation means
    mu_anchor = np.zeros((2, T, 2))
    for k, s in enumerate(OUT):
        lg = panel.logit[s]
        for a in (0, 1):
            for t in range(T):
                cells = lg[(arm_idx == a), t]
                cells = cells[np.isfinite(cells)]
                mu_anchor[a, t, k] = cells.mean() if cells.size else 0.0
    return dict(counts=counts, mask=mask, n_trials=n_trials, arm_idx=arm_idx, age=age,
                mu_anchor=mu_anchor, N=counts.shape[0], T=T)

# ----------------------------------------------------------------------------- simulator
def simulate(true, design, rng):
    N, T = design["N"], design["T"]
    arm, age, ntr = design["arm_idx"], design["age"], design["n_trials"]
    mu = true["mu"]  # (2,T,2)
    sdR, rR = true["sd_RI"], true["r_RI"]
    sdw, rw = true["sd_w1"], true["r_w1"]
    aL, aW, be, de = true["alpha_L"], true["alpha_W"], true["beta"], true["delta"]
    sdz, kap, dage = true["sd_zeta"], true["kappa"], true["d_age"]
    def corr2(sd, r, z):
        x0 = sd[0] * z[:, 0]
        x1 = sd[1] * (r * z[:, 0] + np.sqrt(1 - r * r) * z[:, 1])
        return np.stack([x0, x1], axis=1)
    RI = corr2(sdR, rR, rng.standard_normal((N, 2)))
    w = np.zeros((N, T, 2))
    w[:, 0, :] = corr2(sdw, rw, rng.standard_normal((N, 2)))
    for j in range(1, T):
        z = rng.standard_normal((N, 2)) * sdz
        w[:, j, 0] = aL * w[:, j-1, 0] + be * w[:, j-1, 1] + z[:, 0]
        w[:, j, 1] = aW * w[:, j-1, 1] + de * w[:, j-1, 0] + z[:, 1]
    theta = mu[arm] + dage[None, None, :] * age[:, :, None] + RI[:, None, :] + w  # (N,T,2)
    p = 1.0 / (1.0 + np.exp(-theta))
    p = np.clip(p, EPS, 1 - EPS)
    counts = np.zeros((N, T, 2), dtype=np.int64)
    for k in range(2):
        a = p[:, :, k] * kap[k]
        b = (1 - p[:, :, k]) * kap[k]
        pb = rng.beta(a, b)
        counts[:, :, k] = rng.binomial(ntr[k], pb)
    return counts

# ----------------------------------------------------------------------------- model
def build_riclpm_model(counts, mask, n_trials, arm_idx, age, mu_anchor):
    N, T, K = counts.shape
    assert K == 2
    obs = {k: np.where(mask[:, :, k]) for k in range(2)}  # (rows,cols) per domain
    coords = {"child": np.arange(N), "wave": np.arange(1, T + 1),
              "outcome": ["L", "W"], "arm": ["immediate", "waitlist"], "trans": np.arange(1, T)}
    anchor = mu_anchor  # (2,T,2)
    with pm.Model(coords=coords) as m:
        mu = pm.Normal("mu", mu=anchor, sigma=1.5, dims=("arm", "wave", "outcome"))
        d_age = pm.Normal("d_age", 0.0, 0.5, dims="outcome")
        # correlated random intercepts (2D manual Cholesky)
        sd_RI = pm.HalfNormal("sd_RI", 1.0, dims="outcome")
        r_RI = pm.Deterministic("r_RI", 2.0 * pm.Beta("r_RI_b", 2.0, 2.0) - 1.0)
        zRI = pm.Normal("zRI", 0.0, 1.0, dims=("child", "outcome"))
        RI_L = sd_RI[0] * zRI[:, 0]
        RI_W = sd_RI[1] * (r_RI * zRI[:, 0] + pt.sqrt(1 - r_RI**2) * zRI[:, 1])
        RI = pt.stack([RI_L, RI_W], axis=1)  # (N,2)
        # initial within-person deviation (2D)
        sd_w1 = pm.HalfNormal("sd_w1", 1.0, dims="outcome")
        r_w1 = pm.Deterministic("r_w1", 2.0 * pm.Beta("r_w1_b", 2.0, 2.0) - 1.0)
        zw1 = pm.Normal("zw1", 0.0, 1.0, dims=("child", "outcome"))
        w1_L = sd_w1[0] * zw1[:, 0]
        w1_W = sd_w1[1] * (r_w1 * zw1[:, 0] + pt.sqrt(1 - r_w1**2) * zw1[:, 1])
        # AR + cross-lag
        aL = pm.Normal("alpha_L", 0.0, 0.3)
        aW = pm.Normal("alpha_W", 0.0, 0.3)
        beta = pm.Normal("beta_W_to_L", 0.0, 0.3)
        delta = pm.Normal("delta_L_to_W", 0.0, 0.3)
        pm.Deterministic("delta_minus_beta", delta - beta)
        sd_z = pm.HalfNormal("sd_zeta", 1.0, dims="outcome")
        zzeta = pm.Normal("zzeta", 0.0, 1.0, dims=("child", "trans", "outcome"))
        wL = [w1_L]
        wW = [w1_W]
        for j in range(T - 1):
            zt = zzeta[:, j, :] * sd_z
            Lp, Wp = wL[-1], wW[-1]  # previous-wave deviations, captured before append
            wL.append(aL * Lp + beta * Wp + zt[:, 0])
            wW.append(aW * Wp + delta * Lp + zt[:, 1])
        w = pt.stack([pt.stack(wL, axis=1), pt.stack(wW, axis=1)], axis=2)  # (N,T,2)
        mu_it = mu[arm_idx]  # (N,T,2)
        theta = mu_it + d_age[None, None, :] * age[:, :, None] + RI[:, None, :] + w
        kap = [_priors.kappa_prior().to_pymc("kappa_L"), _priors.kappa_prior().to_pymc("kappa_W")]
        for k in range(2):
            r, c = obs[k]
            beta_binomial_from_logit(
                f"y_{['L','W'][k]}", theta[:, :, k][r, c],
                n_trials=int(n_trials[k]), kappa=kap[k],
                observed=counts[:, :, k][r, c],
            )
    return m

# ----------------------------------------------------------------------------- harness
def true_params(design):
    return dict(
        mu=design["mu_anchor"],                       # realistic arm x wave x out means
        sd_RI=np.array([0.8, 0.8]), r_RI=0.6,         # strong, correlated "ability travels together"
        sd_w1=np.array([0.4, 0.4]), r_w1=0.3,
        alpha_L=0.2, alpha_W=0.2,
        beta=0.05, delta=0.15,                        # true delta - beta = +0.10 (L->W dominant)
        sd_zeta=np.array([0.4, 0.4]),
        kappa=np.array([30.0, 30.0]),
        d_age=np.array([0.0, 0.0]),
    )

def fit_once(counts, design, draws, tune, chains, seed):
    m = build_riclpm_model(counts, design["mask"], design["n_trials"],
                           design["arm_idx"], design["age"], design["mu_anchor"])
    with m:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                          nuts_sampler="nutpie", target_accept=0.9, random_seed=seed,
                          progressbar=False, compute_convergence_checks=False)
    return idata

def _row(idata):
    P = idata.posterior
    d = np.asarray(P["delta_L_to_W"]).ravel()
    b = np.asarray(P["beta_W_to_L"]).ravel()
    dm = np.asarray(P["delta_minus_beta"]).ravel()
    ndiv = int(np.asarray(idata.sample_stats["diverging"]).sum())

    def q(x):
        return np.quantile(x, [0.05, 0.5, 0.95])

    return dict(
        d=q(d).tolist(), b=q(b).tolist(), dm=q(dm).tolist(),
        p_d_gt_b=float((dm > 0).mean()),
        sd_RI=[float(np.asarray(P["sd_RI"])[..., 0].mean()), float(np.asarray(P["sd_RI"])[..., 1].mean())],
        sd_zeta=[float(np.asarray(P["sd_zeta"])[..., 0].mean()), float(np.asarray(P["sd_zeta"])[..., 1].mean())],
        r_RI=float(np.asarray(P["r_RI"]).mean()),
        ndiv=ndiv,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "study"], default="smoke")
    ap.add_argument("--n-sims", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    design = real_design()
    TRUE = true_params(design)
    true_dm = TRUE["delta"] - TRUE["beta"]
    print(f"design: N={design['N']} T={design['T']} n_trials={design['n_trials'].tolist()} "
          f"arms={np.bincount(design['arm_idx']).tolist()} true delta-beta={true_dm:+.3f}", flush=True)
    if a.mode == "smoke":
        rng = np.random.default_rng(12345)
        counts = simulate(TRUE, design, rng)
        t0 = time.time()
        idata = fit_once(counts, design, draws=800, tune=800, chains=4, seed=1)
        r = _row(idata)
        print(f"smoke fit {time.time()-t0:.0f}s  ndiv={r['ndiv']}", flush=True)
        print(f"  delta   (true {TRUE['delta']:+.2f}): {r['d'][1]:+.3f} [{r['d'][0]:+.3f},{r['d'][2]:+.3f}]")
        print(f"  beta    (true {TRUE['beta']:+.2f}): {r['b'][1]:+.3f} [{r['b'][0]:+.3f},{r['b'][2]:+.3f}]")
        print(f"  d-b     (true {true_dm:+.2f}): {r['dm'][1]:+.3f} [{r['dm'][0]:+.3f},{r['dm'][2]:+.3f}]  P(d>b)={r['p_d_gt_b']:.3f}")
        print(f"  sd_RI   (true {TRUE['sd_RI'].tolist()}): {[round(x,2) for x in r['sd_RI']]}")
        print(f"  sd_zeta (true {TRUE['sd_zeta'].tolist()}): {[round(x,2) for x in r['sd_zeta']]}")
        print(f"  r_RI    (true {TRUE['r_RI']}): {r['r_RI']:.2f}")
    else:
        rows = []
        for s in range(a.n_sims):
            rng = np.random.default_rng(1000 + s)
            counts = simulate(TRUE, design, rng)
            t0 = time.time()
            try:
                idata = fit_once(counts, design, draws=500, tune=600, chains=4, seed=2000 + s)
                r = _row(idata)
                r["ok"] = True
            except Exception as e:
                r = {"ok": False, "err": str(e)[:200]}
            r["sim"] = s
            r["secs"] = round(time.time() - t0, 1)
            rows.append(r)
            cov = "?" if not r.get("ok") else ("Y" if r["dm"][0] <= true_dm <= r["dm"][2] else "n")
            print(f"[{s+1}/{a.n_sims}] {r['secs']}s ndiv={r.get('ndiv','-')} "
                  f"dm={r['dm'][1]:+.3f}[{r['dm'][0]:+.3f},{r['dm'][2]:+.3f}] cover90={cov}"
                  if r.get("ok") else f"[{s+1}/{a.n_sims}] FAILED {r.get('err')}", flush=True)
        # summary
        good = [r for r in rows if r.get("ok")]
        if good:
            dm_med = np.array([r["dm"][1] for r in good])
            cover = np.mean([r["dm"][0] <= true_dm <= r["dm"][2] for r in good])
            power = np.mean([r["dm"][0] > 0 for r in good])          # 90% CI excludes 0 (positive)
            pdir = np.mean([r["p_d_gt_b"] > 0.9 for r in good])       # P(d>b)>0.9
            sdRI = np.mean([r["sd_RI"] for r in good], axis=0)
            sdz = np.mean([r["sd_zeta"] for r in good], axis=0)
            ndiv = np.mean([r["ndiv"] for r in good])
            print("\n===== SUMMARY =====", flush=True)
            print(f"  n_ok={len(good)}/{len(rows)}  mean_ndiv={ndiv:.1f}")
            print(f"  delta-beta: true={true_dm:+.3f}  mean_median={dm_med.mean():+.3f}  bias={dm_med.mean()-true_dm:+.3f}  sd_across_sims={dm_med.std():.3f}")
            print(f"  90% CI coverage of (delta-beta): {cover:.2f}  (nominal 0.90)")
            print(f"  power: P(90% CI excludes 0, positive) = {power:.2f};  P(P(d>b)>0.9) = {pdir:.2f}")
            print(f"  variance recovery: mean sd_RI={sdRI.round(2).tolist()} (true {TRUE['sd_RI'].tolist()}); "
                  f"mean sd_zeta={sdz.round(2).tolist()} (true {TRUE['sd_zeta'].tolist()})")
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"true_dm": true_dm, "rows": rows}, f, indent=2)
            print(f"  wrote {a.out}", flush=True)

